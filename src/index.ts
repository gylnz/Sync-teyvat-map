// @ts-ignore
import { app, BrowserWindow, dialog, session, shell, screen } from 'electron'
import * as Log from 'electron-log'
import Store from 'electron-store';
import { autoUpdater } from 'electron-updater'
import { Worker } from 'worker_threads'
import { spawn, ChildProcessWithoutNullStreams } from 'child_process'
import * as process from "process";
import BrowserWindowConstructorOptions = Electron.BrowserWindowConstructorOptions;

const store = new Store()
Log.transports.file.level = 'info'
autoUpdater.logger = Log

const homePath = __dirname.split('dist')[0]
const resourcesPath = homePath.split('app')[0]

class ServerProcess {
  proc: ChildProcessWithoutNullStreams | undefined;

  constructor() {
  }

  start() {
    this.proc = spawn(homePath + 'python\\venv\\Scripts\\python.exe',
        [homePath + 'python\\server.py'], {cwd: homePath + 'python'});
    this.proc.stdout.on('data', (data: Buffer) => {
      let str = data.toString().replace(/\r?\n$/, "")
      console.log(str);
    });
    this.proc.stderr.on('data', (data: Buffer) => {
      let str = data.toString().replace(/\r?\n$/, "")
      console.error(str);
    });
  }

  stop() {
    if (this.proc) {
      this.proc.kill()
    }
  }
}

const server = new ServerProcess()
server.start()

const deploy = async () => {
  await session.defaultSession.loadExtension(resourcesPath + 'extensions', {
    allowFileAccess: true,
  })
  let options: BrowserWindowConstructorOptions = {
    show: false,
    title: `パイモンが地図動かしてくれるヤツ ver.${app.getVersion()}`,
  }

  // @ts-ignore
  let bounds:{x:number, y:number, width:number, height:number} = store.get('winBounds')
  console.log(bounds)
  Object.assign(options, bounds)

  const win = new BrowserWindow(options)
  void win.loadURL('https://act.hoyolab.com/ys/app/interactive-map/index.html?lang=en-us#/map/2?shown_types=&center=2798.50,-3528.00&zoom=0.00')
  win.on('ready-to-show', () => {
    win.show()
  })
  win.on('close', () => {
    let bounds = win.getBounds()
    console.log(bounds)
    if (bounds.x > 1900) {
      bounds.width = bounds.width * 0.8
      bounds.height = bounds.height * 0.8
      console.log(bounds)
    }
    store.set('winBounds', bounds)
  })
  win.on('page-title-updated', (evt) => {
    evt.preventDefault()
  })
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/rollphes/.exec(url)) {
      void shell.openExternal(url)
      return {
        action: 'deny',
      }
    }
    return {
      action: 'allow',
    }
  })
  //win.setMenu(null)
  win.webContents.openDevTools()

  void autoUpdater.checkForUpdatesAndNotify()
  autoUpdater.on('update-downloaded', (info) => {
    void dialog
      .showMessageBox(win, {
        type: 'info',
        buttons: ['更新して再起動', 'あとで'],
        message: 'アップデート',
        detail:
          '新しいバージョンをダウンロードしました。再起動して更新を適用しますか？',
      })
      .then((returnValue) => {
        if (returnValue.response === 0) {
          autoUpdater.quitAndInstall()
        }
      })
  })
}

app.on('ready', () => {
  void deploy()
})

app.on('browser-window-created', (e, win) => {
  win.on('page-title-updated', (evt) => {
    evt.preventDefault()
  })
  win.setMenu(null)
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
    server.stop()
  }
})
