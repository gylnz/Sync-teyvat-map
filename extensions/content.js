const sock = new WebSocket('ws://localhost:27900')

const playImgURL = chrome.runtime.getURL('img/play.png')
const stopImgURL = chrome.runtime.getURL('img/stop.png')
const twitterIconURL = chrome.runtime.getURL('img/twitter-icon.png')
const pinpointImgURL = chrome.runtime.getURL('img/pinpoint.png')
let pauseSyncFlag = true
let oldDimension = "2"
let updateInterval = 4000
let intervalId = setInterval(() => {}, 1000)

const setUp = () =>{
  setTimeout(() => {
    $("div.mhy-map__action-btn--feedback").remove()
    $("div.mhy-map__action-btns").append($.parseHTML(`
      <div id="user-guide-sync" class="mhy-map__action-btn mhy-map__action-btn--routes toggleSync">
      <img src="${pauseSyncFlag?playImgURL:stopImgURL}" class="action-btn__btn-pic">
      <div class="tooltip tooltip--left">Auto Switch</div>
    </div>`))
    $("div.toggleSync").on('click', ()=> {
      pauseSyncFlag = pauseSyncFlag?false:true
      $("div.toggleSync").children("img").attr('src',pauseSyncFlag?playImgURL:stopImgURL)
      if(pauseSyncFlag){
        clearInterval(intervalId)
      }else{
        intervalId = setInterval(() => {
          sock.send(location.hash.slice(6, 7))
        }, updateInterval)
      }
    })
    $("div.mhy-map__action-btns").append($.parseHTML(`
      <div id="user-guide-sync" class="mhy-map__action-btn mhy-map__action-btn--routes Pinpoint">
      <img src="${pinpointImgURL}" class="action-btn__btn-pic">
      <div class="tooltip tooltip--left">Match current location</div>
    </div>`))
    $("div.Pinpoint").on('click', ()=> {
      sock.send(location.hash.slice(6, 7))
    })
    // $("div.mhy-map__action-btns").append($.parseHTML(`
    //   <div id="user-guide-sync" class="mhy-map__action-btn mhy-map__action-btn--routes DeveloperTwitter">
    //   <img src="${twitterIconURL}" class="action-btn__btn-pic">
    //   <div class="tooltip tooltip--left">開発者のTwitter</div>
    // </div>`))
    // $("div.DeveloperTwitter").on('click', ()=> {
    //   window.open('https://twitter.com/rollphes')
    // })
  }, 2000)
}
setUp()

setInterval(()=>{
  $('a[href*="postList"]').remove()
  $("div.bbs-qr").remove()
  const thisDimension = location.hash.slice(6, 7)
  if(thisDimension != oldDimension)setUp()
  oldDimension = thisDimension
},1000)

sock.addEventListener('open', (e) => {
  if (pauseSyncFlag) return
   intervalId = setInterval(() => {
    sock.send(location.hash.slice(6, 7))
   }, updateInterval)
})

sock.addEventListener('message', (e) => {
  if (!e.data.match(/center/g)||!location.hash.match(/center/g)) return
  location.hash = location.hash.replace(/center=.*(?=&)/,e.data)
})

sock.addEventListener('close', (e) => {
  console.log(e)
})

sock.addEventListener('error', (e) => {
  console.log(e)
})
