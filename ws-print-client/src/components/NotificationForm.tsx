import styled, { css } from "styled-components";
import React, { ChangeEventHandler, useContext, useRef, useState } from "react";
import { observer, useLocalObservable } from "mobx-react-lite";
import OneSignal from "react-onesignal";
import { InferContext } from "../main";

const sharedLabel = css`
    font-size: 1.5rem;
`;

const NotificationFormBase = styled.form`
    display: flex;
    flex-direction: column;

    background-color: #ddd;
    padding: 5%;
`;
const FormHeader = styled.h2``;

const FormMiniHeader = styled.h3``;
const FormOptionRow = styled.div`
    display: flex;
    align-items: center;
    flex-direction: row;
    ${sharedLabel}
`;
const FormOptionLabel = styled.label``;
const FormCheckboxBase = styled.input`
    width: 30px;
    height: 30px;
    margin-right: 10px;
`;
const FormInput = styled.input`
    font-size: 1.5rem;
`;
const FormSelectionRow = styled.div`
    display: flex;
    flex-direction: row;
    align-items: center;
    ${sharedLabel}
    justify-content: space-between;
`;
const FormSelectionCheckArea = styled.span`
    display: flex;
`;
const FormSelectionClass = styled.div``;
const FormSelectionInput = styled(FormInput)`
    flex-shrink: 1;
    width: 30%;
`;
const FormSubmit = styled.button`
    margin-top: 10px;
    height: 30px;
    border-radius: 20px;
    outline: none;
    box-shadow: none;
    border: 1px solid black;
`;
interface INotificationFormProps {}

interface INotificationFormCheckboxProps {
    onChange: ChangeEventHandler<HTMLInputElement>;
}
const FormCheckbox = (
    props: INotificationFormCheckboxProps &
        React.DetailedHTMLProps<
            React.InputHTMLAttributes<HTMLInputElement>,
            HTMLInputElement
        >,
) => {
    return (
        <FormCheckboxBase
            {...props}
            onChange={props.onChange}
            type="checkbox"
        />
    );
};

const NotificationForm = observer((props: INotificationFormProps) => {
    const inferStore = useContext(InferContext);

    const onFormConfirm = async (e: SubmitEvent) => {
        e.stopPropagation();
        e.preventDefault();

        await OneSignal.User.PushSubscription.optIn();
        await OneSignal.Slidedown.promptPush();
        if (formState.email != null && formState.email !== "") {
            OneSignal.User.addEmail(formState.email);
        }

        inferStore.notificationsEnabled = true;
        inferStore.sendNotificationRequest(formState);
    };
    const formRef = useRef(null);

    const formState = useLocalObservable(() => ({
        toggleWeb() {
            this.notifications.web = !this.notifications.web;
        },
        toggleEmail() {
            this.notifications.email = !this.notifications.email;
        },
        notifications: {
            web: false,
            email: false,
        },
        email: "",
        toggleStringing() {
            this.alerts.stringing.active = !this.alerts.stringing.active;
        },

        toggleSpaghetti() {
            this.alerts.spaghetti.active = !this.alerts.spaghetti.active;
        },
        setStringingThreshold(e: number) {
            this.alerts.stringing.threshold = e;
        },
        setSpaghettiThreshold(e: number) {
            this.alerts.spaghetti.threshold = e;
        },
        alerts: {
            stringing: {
                active: false,
                threshold: null,
            },
            spaghetti: {
                active: false,
                threshold: null,
            },
        },
    }));

    return (
        <NotificationFormBase ref={formRef} onSubmit={onFormConfirm}>
            <FormHeader>Notifications</FormHeader>
            (Please disable your AdBlocker for notifications to work.)
            <FormOptionRow>
                <FormCheckbox
                    onChange={formState.toggleWeb}
                    checked={formState.notifications.web}
                />
                <FormOptionLabel>Web Notifications</FormOptionLabel>
            </FormOptionRow>
            <FormOptionRow>
                <FormCheckbox
                    onChange={formState.toggleEmail}
                    checked={formState.notifications.email}
                />
                <FormOptionLabel>Email Notifications</FormOptionLabel>
            </FormOptionRow>
            <FormInput placeholder="Enter email here..." />
            <FormMiniHeader>Alerts</FormMiniHeader>
            <FormSelectionRow>
                <FormSelectionCheckArea>
                    <FormCheckbox
                        checked={formState.alerts.stringing.active}
                        onChange={formState.toggleStringing}
                    />
                    <FormOptionLabel> stringing </FormOptionLabel>
                </FormSelectionCheckArea>
                <FormSelectionInput
                    type="number"
                    placeholder="80"
                    max={100}
                    min={0}
                    onChange={(e) =>
                        formState.setStringingThreshold(e.target.valueAsNumber)
                    }
                />
            </FormSelectionRow>
            <FormSelectionRow>
                <FormSelectionCheckArea>
                    <FormCheckbox
                        checked={formState.alerts.spaghetti.active}
                        onChange={formState.toggleSpaghetti}
                    />
                    <FormOptionLabel> spahgetti </FormOptionLabel>
                </FormSelectionCheckArea>
                <FormSelectionInput
                    type="number"
                    max={100}
                    min={0}
                    placeholder="80"
                    onChange={(e) =>
                        formState.setSpaghettiThreshold(e.target.valueAsNumber)
                    }
                />
            </FormSelectionRow>
            <FormSubmit>Confirm</FormSubmit>
        </NotificationFormBase>
    );
});

export default NotificationForm;
