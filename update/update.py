import os
import json

import requests
import asyncio
import aiohttp
import async_timeout

import zendriver
import pyvirtualdisplay

import numpy as np
import pandas as pd

import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


TIMEOUT = 30
ERR = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABIAQMAAABvIyEEAAAABlBMVEUAAABTU1OoaSf/AAAAAXRSTlMAQObYZgAAAENJREFUeF7tzbEJACEQRNGBLeAasBCza2lLEGx0CxFGG9hBMDDxRy/72O9FMnIFapGylsu1fgoBdkXfUHLrQgdfrlJN1BdYBjQQm3UAAAAASUVORK5CYII='
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Priority': 'i',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'same-origin',
    "Sec-Ch-Ua": '"Not_A Brand";v="99", "Chromium";v="142"',
    'Sec-Ch-Mobile': '?0',
    'Sec-Ch-Platform': 'Linux',
}


###############################################################################
# fetch video list
###############################################################################

async def start_browser(proxy):
    browser_args=[
        '--proxy-server={}'.format(proxy),
        '--ignore-certificate-errors',
    ] if proxy is not None else []

    return await zendriver.start(
        headless=False, no_sandbox=True,
        browser_executable_path=os.environ['CHROME_BIN'],
        browser_args=browser_args + [
            '--window-size=1920,1048',
            '--no-gpu',
        ]
    )


async def fetch_user_videos(browser, user):
    request_id = None
    future_body = asyncio.Future()
    loading_finished_future = asyncio.Future()

    # avoid race condition when response_received is triggered before loading_finished
    async def reset_lff():
        nonlocal loading_finished_future
        loading_finished_future = asyncio.Future()

    async def request_handler(ev):
        if '/api/post/item_list' in ev.request.url:
            nonlocal request_id
            request_id = ev.request_id

    async def loading_finished_handler(ev):
        if ev.request_id == request_id:
            loading_finished_future.set_result(ev)

    async def response_received_handler(ev, tab=None):
        if '/api/post/item_list' in ev.response.url:
            await loading_finished_future

            body, _ = await tab.send(
                zendriver.cdp.network.get_response_body(ev.request_id)
            )

            if len(body):
                future_body.set_result(body)
            else:
                # reset loading future, i expect this not to work
                await reset_lff()

        elif ERR == ev.response.url:
            future_body.set_result(None)

    # setup listenersloading_finished
    browser.main_tab.add_handler(
        zendriver.cdp.network.RequestWillBeSent,
        handler=request_handler
    )
    browser.main_tab.add_handler(
        zendriver.cdp.network.ResponseReceived,
        handler=response_received_handler
    )
    browser.main_tab.add_handler(
        zendriver.cdp.network.LoadingFinished,
        handler=loading_finished_handler
    )

    try:
        tab = await browser.get('https://www.tiktok.com/@{}'.format(user))
        body = await asyncio.wait_for(future_body, timeout=TIMEOUT * 2)
    except asyncio.exceptions.TimeoutError:
        print('[!] account request timeout')
        body = None

    # clean
    browser.main_tab.remove_handlers(
        zendriver.cdp.network.RequestWillBeSent,
        handler=request_handler
    )
    browser.main_tab.remove_handlers(
        zendriver.cdp.network.LoadingFinished,
        handler=loading_finished_handler
    )
    browser.main_tab.remove_handlers(
        zendriver.cdp.network.ResponseReceived,
        handler=response_received_handler
    )

    return body


###############################################################################
# subtitles
###############################################################################

async def do_fetch(url, headers, timeout):
    async with aiohttp.ClientSession() as session:
        async with async_timeout.timeout(timeout):
            async with session.get(url, headers=headers, ssl=False) as response:
                return (
                    await response.text(),
                    response.status,
                )


async def fetch_sub(url, user, vid):
    fn = './data/{}/subs/{}'.format(user, vid)
    dn = os.path.dirname(fn)

    if not os.path.isdir(dn):
        os.makedirs(dn)

    if os.path.isfile(fn):
        return

    headers = {
        **HEADERS,
        'Referer': url,
        'Range': 'bytes=0-',
        'Sec-Fetch-Dest': 'video',
    }

    content, status = await do_fetch(url, headers=headers, timeout=TIMEOUT)
    assert status < 300

    if "WEBVTT" not in content[:32]:
        raise Exception('wtf')

    with open(fn, 'w') as f:
        f.write(content)


async def download_subtitles(user, df_u):
    df_t = df_u[
        df_u['author.uniqueId'] == user
    ]['video.subtitleInfos'].explode().apply(pd.Series)

    if not len(df_t):
        return

    df_t['id'] = df_u['id']
    df_t['duration'] = df_u['video.duration']

    df_t = df_t[df_t['LanguageCodeName'].str.contains('spa').fillna(False)]
    df_t = df_t.set_index('id')
    df_t = df_t[~df_t.index.duplicated()]

    for vid, sub in df_t.iterrows():
        try:
            await fetch_sub(sub['Url'], user, vid)
        except:
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(sub['duration'] / 10)


###############################################################################
# main loop
###############################################################################

COLS = [
    'id',
    'createTime',
    'CategoryType',
    'isAd',
    'shareEnabled',
    'duetEnabled',
    'stitchEnabled',
    'itemCommentStatus',

    'desc',
    'textLanguage',
    'AIGCDescription',
    'creatorAIComment.eligibleVideo',
    'creatorAIComment.hasAITopic',
    'creatorAIComment.notEligibleReason',

    'author.id',
    'author.uniqueId',
    'author.nickname',
    'author.verified',
    'author.privateAccount',

    'music.id',
    'music.title',
    'music.authorName',
    'music.original',
    'music.duration',
    'music.isCopyrighted',

    'video.codecType',
    'video.format',
    'video.videoQuality',
    'video.duration',
    'video.width',
    'video.height',
    'video.size',
    'video.bitrate',
    'video.VQScore',

    'video.volumeInfo.Loudness',
    'video.volumeInfo.Peak',
    'video.claInfo.enableAutoCaption',
    'video.claInfo.hasOriginalAudio',
    'video.claInfo.originalLanguageInfo.languageCode',

    'stats.playCount',
    'stats.diggCount',
    'stats.commentCount',
    'stats.shareCount',
    'stats.collectCount'
]
def update_user(user, df_u):
    fn = './data/{}/posts.csv'.format(user)
    dn = os.path.dirname(fn)

    if not os.path.isdir(dn):
        os.makedirs(dn)

    df_u = df_u[COLS]

    if os.path.isfile(fn):
        df_us = pd.read_csv(fn)
        df_u = pd.concat([df_us, df_u], ignore_index=True)
        df_u = df_u[~df_u['id'].duplicated(keep='first')]

    df_u.sort_values('createTime').to_csv(fn, index=False)


def process_data(data):
    try:
        data_obj = json.loads(data)

        df_u = pd.json_normalize(data_obj['itemList'], sep='.')
        df_u = df_u[~df_u['id'].duplicated(keep='last')]

        return df_u
    except:
        return


async def fetch_users(users, proxy):
    async with await start_browser(proxy) as browser:
        for user in users:
            data = await fetch_user_videos(browser, user)
            df_u = process_data(data)

            if df_u is None:
                continue

            update_user(user, df_u)

            await download_subtitles(user, df_u)
            await asyncio.sleep(5 + np.random.random() * 5)


###############################################################################
# setup proxy
###############################################################################

PROXY_URL = 'https://raw.githubusercontent.com/proxifly/free-proxy-list/refs/heads/main/proxies/countries/US/data.csv'
def get_working_proxy():
    proxy_df = pd.read_csv(PROXY_URL, header=None)
    proxy_df = proxy_df[proxy_df[0].str.startswith('socks')].sample(frac=1)

    for proxy in proxy_df[0].values:
        try:
            req = requests.get(
                'https://www.tiktok.com/',
                verify=False,
                proxies={'https': proxy},
                headers=HEADERS,
                timeout=8,
            )
            req.raise_for_status()
            return proxy
        except:
            continue


###############################################################################

if __name__ == '__main__':
    # proxy = get_working_proxy()
    # print(proxy)
    proxy = None

    users = pd.read_csv('./users.csv')['user_name']

    with pyvirtualdisplay.Display(visible=0, size=(1920, 1080)):
        zendriver.loop().run_until_complete(
            fetch_users(users.values, proxy)
        )
