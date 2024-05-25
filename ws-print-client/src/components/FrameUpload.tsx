import styled from "styled-components";
import React, { useContext } from "react";
import Files from "react-files";
import { action, runInAction } from "mobx";
import { InferContext } from "../main";

const FrameUploadBase = styled.div`
    display: flex;
    padding: 5%;
    background-color: #48264a;
    color: white;
    justify-content: center;
    align-content: center;
    font-size: 2rem;
    border-radius: 20px;
    font-weight: 300;
    margin-bottom: 25px;
`;
const UploadIcon = styled.span`
    font-size: 48px;
    padding-right: 20px;
`;
const UploadTextArea = styled.div`
    display: flex;
    align-items: center;
`;
const UploadText = styled.span``;
interface IFrameUploadProps {}
const FrameUpload = (props: IFrameUploadProps) => {
    const inferStore = useContext(InferContext);

    return (
        <Files
            type="file"
            clickable
            accept="image/jpg,image/jpeg,image/png"
            onChange={(files) => {
                const file = files[0];
                runInAction(() => {
                    inferStore.file = file;
                });
                inferStore.requestFrameInfer();
            }}
        >
            <FrameUploadBase>
                <UploadTextArea>
                    <UploadIcon className="material-symbols-outlined">
                        upload
                    </UploadIcon>
                    <UploadText>Upload and Infer Frame</UploadText>
                </UploadTextArea>
            </FrameUploadBase>
        </Files>
    );
};

export default FrameUpload;
