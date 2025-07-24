package com.google.mediapipe.proto;

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.GeneratedMessageLite;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.Parser;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/MediaPipeOptionsProto.class */
public final class MediaPipeOptionsProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/MediaPipeOptionsProto$MediaPipeOptionsOrBuilder.class */
    public interface MediaPipeOptionsOrBuilder extends GeneratedMessageLite.ExtendableMessageOrBuilder<MediaPipeOptions, MediaPipeOptions.Builder> {
    }

    private MediaPipeOptionsProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/MediaPipeOptionsProto$MediaPipeOptions.class */
    public static final class MediaPipeOptions extends GeneratedMessageLite.ExtendableMessage<MediaPipeOptions, Builder> implements MediaPipeOptionsOrBuilder {
        private byte memoizedIsInitialized = 2;
        private static final MediaPipeOptions DEFAULT_INSTANCE;
        private static volatile Parser<MediaPipeOptions> PARSER;

        private MediaPipeOptions() {
        }

        public static MediaPipeOptions parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MediaPipeOptions parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MediaPipeOptions parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MediaPipeOptions parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MediaPipeOptions parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MediaPipeOptions parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MediaPipeOptions parseFrom(InputStream input) throws IOException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MediaPipeOptions parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MediaPipeOptions parseDelimitedFrom(InputStream input) throws IOException {
            return (MediaPipeOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static MediaPipeOptions parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MediaPipeOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MediaPipeOptions parseFrom(CodedInputStream input) throws IOException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MediaPipeOptions parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MediaPipeOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return (Builder) DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(MediaPipeOptions prototype) {
            return (Builder) DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/MediaPipeOptionsProto$MediaPipeOptions$Builder.class */
        public static final class Builder extends GeneratedMessageLite.ExtendableBuilder<MediaPipeOptions, Builder> implements MediaPipeOptionsOrBuilder {
            private Builder() {
                super(MediaPipeOptions.DEFAULT_INSTANCE);
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new MediaPipeOptions();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001��", null);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<MediaPipeOptions> parser = PARSER;
                    if (parser == null) {
                        synchronized (MediaPipeOptions.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return Byte.valueOf(this.memoizedIsInitialized);
                case SET_MEMOIZED_IS_INITIALIZED:
                    this.memoizedIsInitialized = (byte) (arg0 == null ? 0 : 1);
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            MediaPipeOptions defaultInstance = new MediaPipeOptions();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(MediaPipeOptions.class, defaultInstance);
        }

        public static MediaPipeOptions getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        /* JADX DEBUG: Type inference failed for r0v1. Raw type applied. Possible types: com.google.protobuf.Parser<MessageType>, com.google.protobuf.Parser<com.google.mediapipe.proto.MediaPipeOptionsProto$MediaPipeOptions> */
        public static Parser<MediaPipeOptions> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}