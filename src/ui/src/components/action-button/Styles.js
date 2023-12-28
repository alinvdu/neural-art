import styled from "styled-components";

export const StledButtonWrapper = styled.div`
  display: flex;
  padding: 12px 25px;
  border: 1 px solid white;
  border-radius: 5px;
  color: white;
  align-items: center;
  justify-content: center;
  color: black;

  font-size: 25px;
  background-color: white;
  align-self: center;

  @keyframes animate-border {
    0%,
    100% {
      border-width: 1px;
    }
    50% {
      border-width: 5px;
    }
  }

  &:hover {
    border: 5px solid white;
    cursor: pointer;
    animation: animate-border 750ms infinite;
  }
`;
