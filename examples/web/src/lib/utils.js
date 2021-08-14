const AUDIO_DURATION = 0.08;

export function concatenate(resultConstructor, ...arrays) {
  let totalLength = 0;
  for (const arr of arrays) {
      totalLength += arr.length;
  }
  const result = new resultConstructor(totalLength);
  let offset = 0;
  for (const arr of arrays) {
      result.set(arr, offset);
      offset += arr.length;
  }
  return result;
}


window.socket = null;
window.audioData = Float32Array.of();
window.infoSent = false;

export function transcribe(data, lang, sr, cb) {
    window.socket.onmessage = function (evt) {
        console.log("websocket response:", evt.data)
        cb(JSON.parse(evt.data))
    };
    window.audioData = concatenate(Float32Array, window.audioData, data);
    const sendables = grabStreamableAudioData(sr);

    // send extra info first
    if (!window.infoSent) {
        const info = {
            sr: sr,
            modelId: lang,
        }
        const s = JSON.stringify(info);
        window.socket.send(s);
        window.infoSent = true;
    }

    // then, start transcribing
    for (let i = 0; i < sendables.length; i++) {
        const one = sendables[i];
        // console.log(`[${i}] ws sending ${one.length / sr} secs`);
        const blob = concatenate(Float32Array, one, Float32Array.of([]));
        window.socket.send(blob);
    }
}

function grabStreamableAudioData(sr) {
    let minSamples = Math.floor(sr * AUDIO_DURATION);
    let samples = window.audioData.length;
    let sendables = [];
    // grab adequatly many
    while (samples >= minSamples) {
        const one = window.audioData.slice(0, minSamples);
        // console.log("samples:", one.length);
        sendables.push(one);
        window.audioData = window.audioData.slice(minSamples);
        samples = samples - minSamples;
    }
    return sendables;
}

export function grabCtx() {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

export function grabAudioProcessor(audioCtx, bufferSize) {
  let audioNode = null;
  if (audioCtx.createJavaScriptNode) {
      audioNode = audioCtx.createJavaScriptNode(bufferSize, 1, 1);
  } else if (audioCtx.createScriptProcessor) {
      audioNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);
  } else {
      throw new Error('WebAudio not supported!');
  }
  return audioNode
}