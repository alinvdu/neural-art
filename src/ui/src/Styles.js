import styled from "styled-components";
import backgroundBlackPng from "./background-blue.png";

export const StyledAppComponent = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;

  background-image: url("${backgroundBlackPng}");
  background-size: cover;
  background-attachment: fixed;
  width: 100%;
  color: white;
  position: relative;
  min-height: 100%;
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
  width: 350px;
  height: 350px;
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

export const StyledImage = styled.img`
  box-shadow: #000 0px 0px 20px 3px;
  border-radius: 2px;
  border: 1px solid rgba(255, 255, 255, 0.5);
  margin: 8px;
`;

export const StyledImageWrapper = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-gap: 10px;
  margin-bottom: 50px;
`;

export const StyledUploadEegDataWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

export const StyledFormTitle = styled.p``;

export const StyledUploadForm = styled.input`
  display: none;
`;

export const FileInputWrapper = styled.div`
  display: flex;
  align-items: center;
`;

export const StyledFlexWrapper = styled.div`
  display: flex;
  margin-top: 200px;
`;

export const CustomButton = styled.div`
  border: 1px solid white;
  font-size: 18px;
  padding: 5px 12px;
  cursor: pointer;
  &:hover {
    background-color: white;
    color: black;
  }
`;
export const FileNameDisplay = styled.span`
  margin-left: 10px;
  font-size: 17px;
`;
