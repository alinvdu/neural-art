import "./App.css";
import { useEffect, useRef, useState } from "react";
import { ActionButton } from "./components/action-button/ActionButton";
import Baseline from "./components/baseline/Baseline";
import logoPng from "./logo.png";
import CountdownTimer from "./components/countdown-timer/CountdownTimer";
import {
  StyledAppComponent,
  StyledErrorWrapper,
  StyledLogo,
  StyledLogoWrapper,
  StyledMessage,
  StyledTitle,
  StyledVisualFeedback,
  StyledVisualFeedbackWrapper,
} from "./Styles";
import visualFeedback from "./vis-feedback.gif";

const NOT_STARTED = "not-started";
const BASELINE_TESTING = "baseline-testing";
export const IMAGINE_INTRO = "imagine-intro";
const IMAGINE_TIME = 8000;
const PROCESSING_IMAGINATION = "processing-imagination";
const IMAGINE_STAGE = "imagine-stage";
const IMAGINE_COUNTDOWN = "imagine-countdown";
const ERROR = "error";

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
    errorRef.current = error;
  }, [error]);

  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  const addEventListeners = (ws, setId) => {
    ws.addEventListener("message", (event) => {
      const data = JSON.parse(event.data);
      if (data.id && !id) {
        setId(data.id);
      }

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
        // ws.send(
        //   JSON.stringify({
        //     action: "STOP_RECORD",
        //   })
        // );
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
                  if (!errorRef.current) {
                    ws.send(
                      JSON.stringify({
                        action: "START_RECORD",
                        id: idRef.current,
                      })
                    );
                  }
                });
              } else {
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
                ws.send(
                  JSON.stringify({
                    action: "MARK_IMAGINE_START",
                    time: Date.now(),
                    id: idRef.current,
                  })
                );

                setTimeout(() => {
                  setProgress(PROCESSING_IMAGINATION);
                  // stop recording
                  ws.send(
                    JSON.stringify({
                      action: "STOP_RECORD",
                      id: idRef.current,
                    })
                  );
                  setIsRecording(false);
                }, IMAGINE_TIME);
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
