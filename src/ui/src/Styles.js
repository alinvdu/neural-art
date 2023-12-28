import styled from "styled-components";
import backgroundBlackPng from "./background-blue.png";

export const StyledAppComponent = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;

  background-image: url("${backgroundBlackPng}");
  background-size: cover;
  width: 100%;
  height: 100%;
  color: white;
`;

export const StyledLogoWrapper = styled.div`
  position: absolute;
  top: 70px;
  left: 50%;
  transform: translateX(-50%);

  display: flex;
  align-items: center;
`;

export const StyledTitle = styled.span`
  font-size: 25px;
  margin-left: 25px;
  letter-spacing: 4px;
`;

export const StyledLogo = styled.img`
  width: 65px;
`;

export const StyledErrorWrapper = styled.div`
  display: flex;
  flex-direction: column;
`;

export const StyledMessage = styled.div`
  font-size: 25px;
  margin-bottom: 10px;
`;

export const StyledVisualFeedbackWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  p {
    margin-bottom: 25px;
  }
`;

export const StyledVisualFeedback = styled.img`
  width: 420px;
  height: 420px;
  border: 2px solid rgba(255, 255, 255, 1);
  border-radius: 50%;

  @keyframes rotate {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  animation: rotate 100s linear infinite;
  opacity: 1;
`;
