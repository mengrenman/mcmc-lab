"""Regenerate the README screenshots in docs/screenshots.

Drives headless Google Chrome through the DevTools protocol (Python standard library only):
each page is loaded at 1920 x 1080, its lab is started and left running for a few seconds,
then the region from the lesson heading to the end of the lab is captured.

Usage (with the app served locally, e.g. `python3 serve.py`):
    python3 tools/capture_screenshots.py [name ...]
Names: home metropolis hmc-race bayes-gaussian ising-scan replica-exchange lattice-gauge.
Set CHROME to the browser binary if it is not in the default macOS location.
"""
import base64, json, os, socket, struct, subprocess, sys, tempfile, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'screenshots')
CHROME = os.environ.get('CHROME', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:8000/')
PORT = 9333

# name, page, section id (None: top of page through the module cards), setup JS, seconds to run
SHOTS = [
    ('home', 'index.html', None, '', 1),
    ('metropolis', 'metropolis.html', 'lab', "setRange('mh-speed', Math.log10(40)); click('mh-run');", 14),
    ('hmc-race', 'hmc-gibbs.html', 'rc-lab', "setRange('rc-speed', 2); click('rc-run');", 8),
    ('bayes-gaussian', 'bayes.html', 'gs-lab', "click('gs-run');", 6),
    ('ising-scan', 'ising.html', 'sc-lab', "await scan('metropolis'); await scan('heatbath'); await scan('wolff');", 1),
    ('replica-exchange', 'optimization.html', 'rx-lab', "setRange('rx-speed', 3); click('rx-run');", 8),
    ('lattice-gauge', 'lattice.html', 'gl-lab', "setRange('gl-speed', 1.8); click('gl-run');", 10),
]

HELPERS = '''
window.setRange = (id, v) => { const el = document.getElementById(id); el.value = v; el.dispatchEvent(new Event('input')); };
window.click = (id) => document.getElementById(id).click();
window.scan = async (alg) => { const s = document.getElementById('sc-alg'); s.value = alg; click('sc-run');
  while (document.getElementById('sc-run').textContent !== 'Run scan') await new Promise(r => setTimeout(r, 200)); };
'''

REGION = '''(() => {
  const sec = document.getElementById(%r), main = document.querySelector('main').getBoundingClientRect();
  const top = (sec.querySelector('h2') || sec).getBoundingClientRect().top + scrollY - 20;
  const bottom = sec.querySelector('.lab').getBoundingClientRect().bottom + scrollY + 20;
  return { x: main.left - 4, y: top, width: main.width + 8, height: bottom - top };
})()'''

HOME_REGION = '''(() => { const m = document.querySelector('main').getBoundingClientRect();
  const c = document.querySelector('.cards').getBoundingClientRect();
  return { x: m.left - 4, y: 0, width: m.width + 8, height: c.bottom + scrollY + 24 }; })()'''


class DevTools:
    """A minimal WebSocket client, enough for request/response DevTools calls."""

    def __init__(self, ws_url):
        host_port, path = ws_url[len('ws://'):].split('/', 1)
        host, port = host_port.split(':')
        self.sock = socket.create_connection((host, int(port)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f'GET /{path} HTTP/1.1\r\nHost: {host_port}\r\nUpgrade: websocket\r\n'
                           f'Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        buf = b''
        while b'\r\n\r\n' not in buf:
            buf += self.sock.recv(4096)
        if b' 101 ' not in buf.split(b'\r\n')[0]:
            raise ConnectionError(buf.split(b'\r\n')[0].decode())
        self.buf = buf.split(b'\r\n\r\n', 1)[1]
        self.next_id = 0

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise ConnectionError('DevTools connection closed')
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _message(self):
        data = b''
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack('>H', self._read(2))[0]
            elif n == 127:
                n = struct.unpack('>Q', self._read(8))[0]
            data += self._read(n)
            if b0 & 0x80:  # final fragment
                return json.loads(data)

    def call(self, method, **params):
        self.next_id += 1
        msg = json.dumps({'id': self.next_id, 'method': method, 'params': params}).encode()
        n, mask = len(msg), os.urandom(4)
        if n < 126:
            header = bytes([0x81, 0x80 | n])
        elif n < 65536:
            header = bytes([0x81, 0x80 | 126]) + struct.pack('>H', n)
        else:
            header = bytes([0x81, 0x80 | 127]) + struct.pack('>Q', n)
        self.sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(msg)))
        while True:
            m = self._message()
            if m.get('id') == self.next_id:
                if 'error' in m:
                    raise RuntimeError(m['error'])
                return m['result']

    def js(self, expr):
        r = self.call('Runtime.evaluate', expression=expr, awaitPromise=True, returnByValue=True)
        if 'exceptionDetails' in r:
            raise RuntimeError(r['exceptionDetails'])
        return r['result'].get('value')


def main():
    only = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    profile = tempfile.mkdtemp(prefix='mcmc-lab-chrome-')
    chrome = subprocess.Popen([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars',
                               f'--remote-debugging-port={PORT}', '--remote-allow-origins=*',
                               f'--user-data-dir={profile}', '--window-size=1920,1080', 'about:blank'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = None
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json'))
                page = next(t for t in targets if t['type'] == 'page')
                break
            except Exception:
                time.sleep(0.2)
        if not page:
            raise RuntimeError('Chrome did not start')
        dt = DevTools(page['webSocketDebuggerUrl'])
        dt.call('Page.enable')
        dt.call('Emulation.setDeviceMetricsOverride', width=1920, height=1080, deviceScaleFactor=1, mobile=False)
        dt.call('Emulation.setFocusEmulationEnabled', enabled=True)  # keep animation frames flowing
        dt.call('Page.bringToFront')
        for name, url, sec, setup, secs in SHOTS:
            if only and name not in only:
                continue
            dt.call('Page.navigate', url=BASE + url)
            time.sleep(0.5)
            for _ in range(60):  # wait until the page's modules have run and the math is typeset
                if dt.js("document.readyState === 'complete' && !!document.querySelector('.site-header') && !!document.querySelector('main .katex')"):
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(f'{url} never became ready')
            time.sleep(0.8)
            dt.js(HELPERS)
            if sec:  # off-screen labs pause, so bring this one into view while it runs
                dt.js(f"document.getElementById({sec!r}).querySelector('.lab').scrollIntoView({{block: 'center'}})")
            dt.js(f'(async () => {{ {setup} }})()')
            time.sleep(secs)
            if sec:
                dt.js("document.querySelectorAll('button.btn.primary').forEach(b => { if (b.textContent === 'Pause') b.click(); })")
                dt.js('window.scrollTo(0, 0)')  # keeps the sticky header above the captured region
                time.sleep(0.6)
            clip = dt.js(REGION % sec if sec else HOME_REGION)
            clip['scale'] = 1
            shot = dt.call('Page.captureScreenshot', format='png', clip=clip, captureBeyondViewport=True)
            path = os.path.join(OUT, name + '.png')
            with open(path, 'wb') as fh:
                fh.write(base64.b64decode(shot['data']))
            print(f'{name}: {int(clip["width"])} x {int(clip["height"])}, {os.path.getsize(path) // 1024} KB')
    finally:
        chrome.terminate()
        try:
            chrome.wait(5)
        except subprocess.TimeoutExpired:
            chrome.kill()


if __name__ == '__main__':
    main()
