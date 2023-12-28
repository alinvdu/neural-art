import React, { useState, useEffect } from "react";
import styled from "styled-components";

const StyledTimerWrapper = styled.div`
  font-size: 75px;
  font-weight: bold;
  @keyframes fade {
    0%,
    100% {
      opacity: 0;
    }
    50% {
      opacity: 1;
    }
  }

  animation: fade 1s infinite;
`;

const CountdownTimer = ({ initialCount, onFinish }) => {
  const [count, setCount] = useState(initialCount);

  useEffect(() => {
    const interval = setInterval(() => {
      setCount((prevCount) => (prevCount > 0 ? prevCount - 1 : 0));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (count === 0) {
      onFinish();
    }
  }, [count]);

  return <StyledTimerWrapper>{count}</StyledTimerWrapper>;
};

export default CountdownTimer;
