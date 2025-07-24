package com.google.mediapipe.framework;

import com.google.common.base.Charsets;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/MediaPipeException.class */
public class MediaPipeException extends RuntimeException {
    private final StatusCode statusCode;
    private final String statusMessage;

    public MediaPipeException(int statusCode, String statusMessage) {
        super(StatusCode.values()[statusCode].description() + ": " + statusMessage);
        this.statusCode = StatusCode.values()[statusCode];
        this.statusMessage = statusMessage;
    }

    MediaPipeException(int code, byte[] message) {
        this(code, new String(message, Charsets.UTF_8));
    }

    public StatusCode getStatusCode() {
        return this.statusCode;
    }

    public String getStatusMessage() {
        return this.statusMessage;
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/MediaPipeException$StatusCode.class */
    public enum StatusCode {
        OK("ok"),
        CANCELLED("canceled"),
        UNKNOWN("unknown"),
        INVALID_ARGUMENT("invalid argument"),
        DEADLINE_EXCEEDED("deadline exceeded"),
        NOT_FOUND("not found"),
        ALREADY_EXISTS("already exists"),
        PERMISSION_DENIED("permission denied"),
        RESOURCE_EXHAUSTED("resource exhausted"),
        FAILED_PRECONDITION("failed precondition"),
        ABORTED("aborted"),
        OUT_OF_RANGE("out of range"),
        UNIMPLEMENTED("unimplemented"),
        INTERNAL("internal"),
        UNAVAILABLE("unavailable"),
        DATA_LOSS("data loss"),
        UNAUTHENTICATED("unauthenticated");
        
        private final String description;

        StatusCode(String description) {
            this.description = description;
        }

        public String description() {
            return this.description;
        }
    }
}