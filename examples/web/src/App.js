import React from 'react';

import './App.css';
import { transcribe, grabAudioProcessor, grabCtx } from './lib/utils';


// dev
// const URI_BASE = "localhost:8080";

// prod
const URI_BASE = window.location.host;

const URI_WEBSOCKET = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${URI_BASE}/websocket`;
const URI_API = `${window.location.protocol}//${URI_BASE}/api`;


class App extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      recording: false,
      connected: false,
      error: "",
      emoji: "glyphicon glyphicon-option-horizontal",
      sampleRate: "...",
      bufferLength: "...",
      selectedLanguage: "invalid",
      languagesAvailable: [],
      apiInfo: {
        "languages": [],
      },
      allowRecording: false,
    };
    this.handleClickRecordStream = this.handleClickRecordStream.bind(this);
    this.handleLanguageChange = this.handleLanguageChange.bind(this);
    this.processor = null;
  }

  componentDidMount() {
    if (!window.socket) {
        try {
          var socket = new WebSocket(URI_WEBSOCKET)
        } catch (err) {
          this.setState({
            connected: false,
            error: err.type,
            emoji: "glyphicon glyphicon-flash text-danger",
          })
          return;
        }
        socket.onopen = (function() {
          this.setState({
            connected: true,
            error: "connected",
            emoji: "glyphicon glyphicon-ok text-success",
          })
          console.log("ws opened.");
        }).bind(this);
        socket.onerror = (function(err) {
          this.setState({
            connected: false,
            error: err.type,
            emoji: "glyphicon glyphicon-flash text-danger",
          })
          console.log("ws error:", err);
        }).bind(this);
        window.socket = socket;
    }

    // fetch api info stuff
    //  and auto-choose lang
    fetch(`${URI_API}`)
      .then(response => response.json())
      .then((data) => {
        this.setState({
          apiInfo: data,
          languagesAvailable: data.languages
            .filter(lang => lang.enable)
            .map(lang => lang.code),
        });
        this.chooseFirstLang();
      })

  }

  chooseFirstLang() {
    const storedCode = localStorage.getItem("selectedLanguage");
    let obj = {
      target: {
        value: "invalid"
      }
    } 
    if(storedCode) {
      obj.target.value = storedCode;
      this.handleLanguageChange(obj, true);
    } else {
      if(this.state.languagesAvailable.length > 0) {
        obj.target.value = this.state.languagesAvailable[0];
        this.handleLanguageChange(obj, true);
      }
    }
  }

  stopRecording() {
    this.processor.onaudioprocess = null;
    window.audioData = Float32Array.of();
  }

  startRecording() {
    const onAudioProcess = (function (e) {
        // Do something with the data, e.g. convert it to WAV
        const sr = e.inputBuffer.sampleRate;
        const lang = this.state.selectedLanguage;
        const data = e.inputBuffer.getChannelData(0);
        this.setState({
            sampleRate: sr + " Hz",
            bufferLength: e.inputBuffer.length,
        })
        transcribe(data, lang, sr, (transcript) => {
            document.getElementById("transcript").innerText = transcript
        });
    }).bind(this);

    const onMicrophoneCaptured = (function(stream) {
        console.log("stream:", stream);

        const context = grabCtx();
        const source = context.createMediaStreamSource(stream);
        const processor = grabAudioProcessor(context);
        this.processor = processor;

        source.connect(processor);
        processor.connect(context.destination);
        processor.onaudioprocess = onAudioProcess;
    }).bind(this);

    const onMicrophoneError = (function(err) {
        console.error(err);
    });

    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
        .then(onMicrophoneCaptured)
        .catch(onMicrophoneError);
  }

  handleClickRecordStream(e) {
    e.preventDefault();
    if (!this.state.allowRecording) return;
    if (this.state.recording) {
        this.stopRecording();
    } else {
        this.startRecording();
    }
    this.setState((state) => ({
      recording: !state.recording
    }))
  }

  handleLanguageChange(e, connected=true) {
    const code = e.target.value;
    const isConnected = connected ? true : this.state.connected;
    this.setState({
      selectedLanguage: code,
      allowRecording: isConnected &&
        this.state.languagesAvailable.includes(code),
    });
    localStorage.setItem("selectedLanguage", code);
  }

  render() {
    const classes = "btn btn-lg btn-primary";
    const classesDisabled = classes + " disabled";
    const recordingClasses = "recordIcon glyphicon glyphicon-record";
    const recordingClassesRecording = recordingClasses + " recording";
    return (
      <div className="container App">

        <h1>LibreASR</h1>

        <div className="container">
          <div className="info">
            <div className="row">
              <div className="col-xs-5">
                Sample Rate
              </div>
              <div className="col-xs-1">
                {this.state.sampleRate}
              </div>
            </div>

            <div className="row">
              <div className="col-xs-5">
                Buffer Size
              </div>
              <div className="col-xs-1">
                {this.state.bufferLength}
              </div>
            </div>

            <div className="row">
              <div className="col-xs-5">
                Websocket Status
              </div>
              <div className="col-xs-4">
                <div id="ws-status">
                  {this.state.connected}{this.state.error}
                  &nbsp;
                  <span className={this.state.emoji}></span>
                </div>
              </div>
            </div>

            <div className="languageSelector">
              Choose Language/Model
            <form>
              {this.state.apiInfo.languages.map((lang, index) => {
                return <div className="radio" key={lang.code}>
                  <label>
                    <input type="radio" value={lang.code} 
                      checked={this.state.selectedLanguage === lang.code} 
                      disabled = {lang.enable ? "" : "disabled"}
                      onChange={this.handleLanguageChange} />
                    {lang.name}
                  </label>
                </div>
              })}
            </form>
            </div>

          </div>
        </div>

        <h2>Transcribe Stream</h2>

        <div className="container">

          <div className="row">
            <div className="transcript well" id="transcript"></div>
          </div>

          <div className="row text-center">
            <span className={this.state.recording ? recordingClassesRecording : recordingClasses}></span>
            &nbsp;
            <button
            className={this.state.allowRecording ? classes : classesDisabled}
            onClick={this.handleClickRecordStream}
            id="record">
              {this.state.recording ? "Stop": "Start"} recording!
            </button>
          </div>

        </div>

      </div>
    );
  }

}

export default App;
