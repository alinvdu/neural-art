import "./App.css";
import { useEffect, useRef, useState } from "react";
import { ActionButton } from "./components/action-button/ActionButton";
import Baseline from "./components/baseline/Baseline";
import logoPng from "./logo.png";
import CountdownTimer from "./components/countdown-timer/CountdownTimer";
import {
  CustomButton,
  FileInputWrapper,
  FileNameDisplay,
  StyledAppComponent,
  StyledErrorWrapper,
  StyledFlexWrapper,
  StyledFormTitle,
  StyledImage,
  StyledImageWrapper,
  StyledLogo,
  StyledLogoWrapper,
  StyledMessage,
  StyledTitle,
  StyledUploadEegDataWrapper,
  StyledUploadForm,
  StyledVisualFeedback,
  StyledVisualFeedbackWrapper,
} from "./Styles";
import visualFeedback from "./vis-feedback.gif";

const NOT_STARTED = "not-started";
const BASELINE_TESTING = "baseline-testing";
export const IMAGINE_INTRO = "imagine-intro";
const IMAGINE_TIME = 8400;
const PROCESSING_IMAGINATION = "processing-imagination";
const IMAGINE_STAGE = "imagine-stage";
const IMAGINE_COUNTDOWN = "imagine-countdown";
const ERROR = "error";
const SHOW_IMAGES = "show-images";
const UPLOAD_EEG_DATA = "upload-eeg-data";

const isEmpty = (myEmptyObj) =>
  Object.keys(myEmptyObj).length === 0 && myEmptyObj.constructor === Object;

function App() {
  const [progress, setProgress] = useState(NOT_STARTED);
  const [id, setId] = useState(null);
  const [ws, setWs] = useState(null);
  const [hasBaseline, setBaseline] = useState(null);
  const [error, setError] = useState(null);
  const errorRef = useRef(error);
  const progressRef = useRef(progress);
  const [isRecording, setIsRecording] = useState(false);
  const idRef = useRef(id);
  const [images, setImages] = useState(null);
  const eegDataInputRef = useRef(null);
  const eegMetadataInputRef = useRef(null);
  const [eegFilename, setEegFileName] = useState("");
  const [metadataFileName, setMetadataFileName] = useState("");
  // const [baselineMarkerId, setBaselineMarkerId] = useState(null);
  // const [imagineMarkerId, setImagineMarkerId] = useState(null);

  useEffect(() => {
    idRef.current = id;
  }, [id]);

  useEffect(() => {
    return () => {
      if (ws) {
        ws.send(
          JSON.stringify({
            action: "STOP_RECORD",
            id: idRef.current,
          })
        );
      }
    };
  }, []);

  useEffect(() => {
    if (progress === IMAGINE_STAGE) {
      const time = Date.now();
      console.log("sending time", time);
      ws.send(
        JSON.stringify({
          action: "MARK_IMAGINE_START",
          id: idRef.current,
          time,
        })
      );

      setTimeout(() => {
        setProgress(UPLOAD_EEG_DATA);
        // stop recording
        ws.send(
          JSON.stringify({
            action: "STOP_RECORD",
            id: idRef.current,
            // markerId: imagineMarkerId,
            // time: Date.now(),
          })
        );
        setIsRecording(false);
      }, IMAGINE_TIME);
    }
  }, [progress]);

  const uploadFile = (file, isEEGData) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      let fileData = {
        isEEGData,
        content: Array.from(new Uint8Array(event.target.result)),
      };
      ws.send(
        JSON.stringify({
          action: "UPLOAD_FILE",
          id: idRef.current,
          fileData,
        })
      );
    };

    reader.readAsArrayBuffer(file);
  };

  useEffect(() => {
    errorRef.current = error;
  }, [error]);

  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  const addEventListeners = (ws, setId) => {
    ws.addEventListener("message", (event) => {
      const data = JSON.parse(event.data);

      switch (data.action) {
        case "RECORD_STARTED":
          console.log("started recording");
          setIsRecording(true);
          if (progressRef.current === IMAGINE_INTRO) {
            setProgress(IMAGINE_COUNTDOWN);
          } else {
            if (!hasBaseline) {
              setProgress(BASELINE_TESTING);
            }
          }
          break;
        case "ERROR":
          setProgress(ERROR);
          setError(
            isEmpty(data.err) && data.err
              ? {
                  message:
                    "An error occurred, check logs and retry! Or open an issue",
                }
              : data.err
          );
          break;
        case "SEND_IMAGES":
          setProgress(SHOW_IMAGES);
          setImages(data.images);
          break;
        // case "SET_BASELINE_MARKER":
        //   setBaselineMarkerId(data.markerId);
        //   break;
        // case "SET_IMAGINE_MARKER_ID":
        //   setImagineMarkerId(data.markerId);
        //   break;
        case "FILE_UPLOADED":
          setProgress(PROCESSING_IMAGINATION);
          break;
        case "SET_ID":
          setId(data.id);
          if (!errorRef.current) {
            console.log("sending start record 1");
            ws.send(
              JSON.stringify({
                action: "START_RECORD",
                id: data.id,
              })
            );
          }
          break;
      }
    });
  };

  const renderContext = () => {
    switch (progress) {
      case NOT_STARTED:
        return (
          <ActionButton
            text="Start eeg record"
            onClick={() => {
              // setup the ws connection if it doesn't exist
              if (!id) {
                const ws = new WebSocket("ws://localhost:9001");
                addEventListeners(ws, setId);
                ws.addEventListener("open", (open) => {
                  setWs(ws);
                  ws.send(
                    JSON.stringify({
                      action: "INITIATE",
                    })
                  );
                });
              } else {
                console.log("sending start record 2");
                ws.send(
                  JSON.stringify({
                    action: "START_RECORD",
                    id: idRef.current,
                  })
                );
              }
            }}
          />
        );
      case BASELINE_TESTING:
        return (
          <Baseline
            setProgress={(curProgress) => {
              if (!errorRef.current) {
                setProgress(curProgress);
              }
            }}
            markBaseline={() => {
              ws.send(
                JSON.stringify({
                  action: "MARK_BASELINE",
                  time: Date.now(),
                  id: idRef.current,
                })
              );
            }}
            // markEndBaseline={() => {
            //   ws.send(
            //     JSON.stringify({
            //       action: "SET_MARKER_END",
            //       markerId: baselineMarkerId,
            //     })
            //   );
            // }}
          />
        );
      case IMAGINE_INTRO:
        return (
          <ActionButton
            text="Start imagining"
            onClick={() => {
              if (!isRecording) {
                ws.send(
                  JSON.stringify({
                    action: "START_RECORD",
                    id: idRef.current,
                  })
                );
              } else {
                setProgress(IMAGINE_COUNTDOWN);
              }
            }}
          />
        );
      case IMAGINE_COUNTDOWN: {
        return (
          <CountdownTimer
            initialCount={5}
            onFinish={() => {
              if (!error) {
                setProgress(IMAGINE_STAGE);
              }
            }}
          />
        );
      }
      case IMAGINE_STAGE:
        return (
          <StyledVisualFeedbackWrapper>
            <p>Recording...</p>
            <StyledVisualFeedback src={visualFeedback} />
          </StyledVisualFeedbackWrapper>
        );
      case PROCESSING_IMAGINATION:
        return <div>Generating images</div>;
      case ERROR:
        return (
          <StyledErrorWrapper>
            <StyledMessage>{error.message}</StyledMessage>
            <ActionButton
              text="Retry"
              onClick={() => {
                setError(null);
                setProgress(IMAGINE_INTRO);
              }}
            />
          </StyledErrorWrapper>
        );
      case SHOW_IMAGES:
        return (
          <StyledFlexWrapper>
            <StyledImageWrapper>
              {images.map((base64Img, index) => (
                <StyledImage
                  key={index}
                  src={`data:image/png;base64,${base64Img}`}
                  alt={`Generated Image ${index + 1}`}
                />
              ))}
            </StyledImageWrapper>
          </StyledFlexWrapper>
        );
      case UPLOAD_EEG_DATA:
        const handleEEGDataButtonClick = () => {
          eegDataInputRef.current.click();
        };
        const handleMetadataInputRef = () => {
          eegMetadataInputRef.current.click();
        };
        return (
          <StyledUploadEegDataWrapper>
            <StyledFormTitle>
              Locate the last record in Emotiv PRO software and export the CSV
              manually
            </StyledFormTitle>
            <StyledFormTitle>Upload Recorded EEG Data</StyledFormTitle>
            <FileInputWrapper>
              <StyledUploadForm
                type="file"
                ref={eegDataInputRef}
                accept=".csv"
                onChange={(event) => {
                  const input = event.target.files[0];
                  setEegFileName(input ? input.name : "");
                  if (input) {
                    uploadFile(input, true);
                  }
                }}
              />
              <CustomButton onClick={handleEEGDataButtonClick}>
                Browse...
              </CustomButton>
              {eegFilename && <FileNameDisplay>{eegFilename}</FileNameDisplay>}
            </FileInputWrapper>
            <StyledFormTitle>Upload Recorded EEG MetaData</StyledFormTitle>
            <FileInputWrapper>
              <StyledUploadForm
                type="file"
                ref={eegMetadataInputRef}
                accept=".csv"
                onChange={(event) => {
                  const input = event.target.files[0];
                  setMetadataFileName(input ? input.name : "");
                  if (input) {
                    uploadFile(input, false);
                  }
                }}
              />
              <CustomButton onClick={handleMetadataInputRef}>
                Browse...
              </CustomButton>
              {metadataFileName && (
                <FileNameDisplay>{metadataFileName}</FileNameDisplay>
              )}
            </FileInputWrapper>
          </StyledUploadEegDataWrapper>
        );
    }
  };

  return (
    <StyledAppComponent>
      <StyledLogoWrapper>
        <StyledLogo src={logoPng} />
        <StyledTitle>Neural Art</StyledTitle>
      </StyledLogoWrapper>
      {renderContext()}
    </StyledAppComponent>
  );
}

export default App;
