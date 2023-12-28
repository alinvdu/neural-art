import React from "react";
import { StledButtonWrapper } from "./Styles";

export const ActionButton = ({ text, onClick }) => {
  return <StledButtonWrapper onClick={onClick}>{text}</StledButtonWrapper>;
};
