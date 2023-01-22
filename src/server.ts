// @ts-ignore
import * as opencvBuild from '@u4/opencv-build'
// @ts-ignore
import * as cv from '@u4/opencv4nodejs'
// @ts-ignore
import { AKAZEDetector, Vec2 } from '@u4/opencv4nodejs'
import * as fs from 'fs'
import screenshot from 'screenshot-desktop'
import { WebSocket, WebSocketServer } from 'ws'

console.log = (function() {
  const console_log = console.log;
  const timeStart = new Date().getTime();

  return function() {
    const delta = new Date().getTime() - timeStart;
    const args = [];
    args.push((delta / 1000).toFixed(2) + ':');
    for(let i = 0; i < arguments.length; i++) {
      args.push(arguments[i]);
    }
    console_log.apply(console, args);
  };
})();


process.env.OPENCV_BIN_DIR = new opencvBuild.OpenCVBuildEnv().opencvBinDir
process.env.path += ';' + new opencvBuild.OpenCVBuildEnv().opencvBinDir

interface MinMatches {
  minDeg: number
  descriptorMatchsI: cv.DescriptorMatch
  descriptorMatchsJ?: cv.DescriptorMatch
}
interface Dimension {
  name: string
  x: number
  y: number
}

const homePath = __dirname.split('dist')[0]
const resourcesPath = homePath.split('app')[0]

const akaze = new AKAZEDetector()
const wss = new WebSocketServer({ port: 27900 })

const dimensions: { [key in number]: Dimension } = {
  2: {
    name: 'map',
    x: 7002,
    y: 1126,
  },
  7: {
    name: 'enkanomiya',
    x: 1861,
    y: 1783,
  },
  9: {
    name: 'sougan',
    x: 2027,
    y: 1958,
  },
}
let clientDimension = '2'

const showCapture = async (image: any) => {
  // cv.imshowAsync("capture", image).then(() => {
  //   cv.waitKey()
  // })
}

const findMap = async (dimension: Dimension) => {
  await screenshot({ filename: resourcesPath + 'img\\t.png' })
  const targetImgOri = await cv.imreadAsync(
    resourcesPath + 'img\\t.png',
    cv.IMREAD_GRAYSCALE
  )
  const region = new cv.Rect(
    Math.floor(targetImgOri.cols * 0.03125),
    Math.floor(targetImgOri.rows * 17 / 1080),
    Math.floor(targetImgOri.rows * 215 / 1080),
    Math.floor(targetImgOri.rows * 215 / 1080)
  )
  const regionImg = targetImgOri.getRegion(region)
  showCapture(regionImg)
  const targetImg = regionImg.resize(600, 600)

  const targetImgKeyPoints = await akaze.detectAsync(targetImg)

  const targetImgDescriptors = await akaze.computeAsync(
    targetImg,
    targetImgKeyPoints
  )
  const mapImgKeyPoints0 = (
    JSON.parse(
      fs.readFileSync(
        resourcesPath + `data\\${dimension.name}ImgKeyPoints.dat`,
        'utf-8'
      )
    ) as cv.KeyPoint[]
  ).map(
    (key) =>
      new cv.KeyPoint(
        new cv.Point2(key.pt.x, key.pt.y),
        key.size,
        key.angle,
        key.response,
        key.octave,
        key.class_id
      )
  )
  const mapImgDescriptors0 = new cv.Mat(
    JSON.parse(
      fs.readFileSync(
        resourcesPath + `data\\${dimension.name}ImgDescriptors.dat`,
        'utf-8'
      )
    ) as number[][],
    0
  )

  const bf = new cv.BFMatcher(cv.NORM_HAMMING2)
  const matches = bf.match(targetImgDescriptors, mapImgDescriptors0)
  if (matches.length === 0) return
  const bestN = 40
  const bestMatches = matches
    .sort((match1: { distance: number }, match2: { distance: number }) => match1.distance - match2.distance)
    .slice(0, bestN)

  const resultMatches: cv.DescriptorMatch[] = []
  for (let i = 0; i < bestMatches.length - 1; i++) {
    const minMatches: MinMatches = {
      minDeg: 360,
      descriptorMatchsI: bestMatches[i],
    }
    for (let j = i + 1; j < bestMatches.length; j++) {
      if (i == j) continue
      const mapRad = Math.atan2(
          (mapImgKeyPoints0[bestMatches[i].trainIdx].pt.y -
          mapImgKeyPoints0[bestMatches[j].trainIdx].pt.y),
          (mapImgKeyPoints0[bestMatches[i].trainIdx].pt.x -
            mapImgKeyPoints0[bestMatches[j].trainIdx].pt.x)
      )
      const targetRad = Math.atan2(
          (targetImgKeyPoints[bestMatches[i].queryIdx].pt.y -
          targetImgKeyPoints[bestMatches[j].queryIdx].pt.y),
          (targetImgKeyPoints[bestMatches[i].queryIdx].pt.x -
            targetImgKeyPoints[bestMatches[j].queryIdx].pt.x)
      )
      let mapDeg = (mapRad * (180 / Math.PI)) % 360
      let targetDeg = (targetRad * (180 / Math.PI)) % 360
      mapDeg = mapDeg < 0 ? 360 + mapDeg : mapDeg
      targetDeg = targetDeg < 0 ? 360 + targetDeg : targetDeg
      const diffDeg = Math.abs(mapDeg - targetDeg)
      if (isNaN(diffDeg)) continue
      if (minMatches.minDeg > diffDeg) {
        minMatches.descriptorMatchsJ = bestMatches[j]
        minMatches.minDeg = diffDeg
      }
    }
    if (minMatches.minDeg > 0.2 || minMatches.descriptorMatchsI.distance > 50)
      continue
    resultMatches.push(minMatches.descriptorMatchsI)
    if (
      minMatches.descriptorMatchsJ === undefined ||
      minMatches.descriptorMatchsJ.distance > 50
    )
      continue
    resultMatches.push(minMatches.descriptorMatchsJ)
  }
  if (resultMatches.length < 2) return
  resultMatches.slice(0, 2)
  const mag =
    new Vec2(
      mapImgKeyPoints0[resultMatches[0].trainIdx].pt.x -
        mapImgKeyPoints0[resultMatches[1].trainIdx].pt.x,
      mapImgKeyPoints0[resultMatches[0].trainIdx].pt.y -
        mapImgKeyPoints0[resultMatches[1].trainIdx].pt.y
    ).norm() /
    new Vec2(
      targetImgKeyPoints[resultMatches[0].queryIdx].pt.x -
        targetImgKeyPoints[resultMatches[1].queryIdx].pt.x,
      targetImgKeyPoints[resultMatches[0].queryIdx].pt.y -
        targetImgKeyPoints[resultMatches[1].queryIdx].pt.y
    ).norm()

  const mapVec2 = new Vec2(
    300 - targetImgKeyPoints[resultMatches[0].queryIdx].pt.x,
    300 - targetImgKeyPoints[resultMatches[0].queryIdx].pt.y
  )
    .mul(mag)
    .add(
      new Vec2(
        mapImgKeyPoints0[resultMatches[0].trainIdx].pt.x,
        mapImgKeyPoints0[resultMatches[0].trainIdx].pt.y
      )
    )
  const { x, y } = mapVec2 as Vec2
  const ret = `center=${Math.floor((y - dimension.y) * 100) / 100},${
    Math.floor((x - dimension.x) * 100) / 100
  }`
  return ret
}

let processing = false
const sendClient = async (ws: WebSocket, dimension: Dimension) => {
  console.log(dimension)
  if (processing) {
    console.log("SKIP")
    return
  }
  processing = true
  findMap(dimension)
      .then((center) => {
          console.log(center)
          if (!center) return
          ws.send(center)
      })
      .catch((e) => {})
      .finally(() => {
        processing = false;
      })
}

wss.on('connection', (ws) => {
  ws.on('message', (e) => {
    clientDimension = e.toString()
    void sendClient(ws, dimensions[+clientDimension])
  })
})
