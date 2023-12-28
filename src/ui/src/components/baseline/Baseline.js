import React, { useEffect, useState } from "react";
import CountdownTimer from "../countdown-timer/CountdownTimer";
import styled from "styled-components";
import { IMAGINE_INTRO } from "../../App";

const StyledFocusPoint = styled.div`
  font-size: 100px;
  font-family: "Roboto", sans-serif;
  font-weight: 100;
`;

const StyledIntroText = styled.p`
  font-size: 25px;
`;

const INTRO_DELAY = 2500;
const BASELINE_TIME = 8000;

const Baseline = ({ setProgress, markBaseline }) => {
  const [phase, setPhase] = useState("INTRO");

  useEffect(() => {
    const timeout = setTimeout(() => {
      setPhase("COUNTDOWN");
    }, INTRO_DELAY);
    return () => {
      clearTimeout(timeout);
    };
  }, []);

  const renderPhase = () => {
    switch (phase) {
      case "INTRO":
        return (
          <StyledIntroText>
            Please relax while baseline is measured
          </StyledIntroText>
        );
      case "COUNTDOWN":
        return (
          <CountdownTimer
            initialCount={5}
            onFinish={() => {
              setPhase("FOCUS");
              markBaseline();
              // transmit the beginning the marking of baseline start
              setTimeout(() => {
                setProgress(IMAGINE_INTRO);
              }, BASELINE_TIME);
            }}
          />
        );
      case "FOCUS":
        return <StyledFocusPoint>+</StyledFocusPoint>;
      default:
        return <StyledFocusPoint>+</StyledFocusPoint>;
    }
  };
  return <div>{renderPhase()}</div>;
};

export default Baseline;
