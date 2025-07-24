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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorOptionsProto.class */
public final class CalculatorOptionsProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorOptionsProto$CalculatorOptionsOrBuilder.class */
    public interface CalculatorOptionsOrBuilder extends GeneratedMessageLite.ExtendableMessageOrBuilder<CalculatorOptions, CalculatorOptions.Builder> {
        @Deprecated
        boolean hasMergeFields();

        @Deprecated
        boolean getMergeFields();
    }

    private CalculatorOptionsProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorOptionsProto$CalculatorOptions.class */
    public static final class CalculatorOptions extends GeneratedMessageLite.ExtendableMessage<CalculatorOptions, Builder> implements CalculatorOptionsOrBuilder {
        private int bitField0_;
        public static final int MERGE_FIELDS_FIELD_NUMBER = 1;
        private boolean mergeFields_;
        private byte memoizedIsInitialized = 2;
        private static final CalculatorOptions DEFAULT_INSTANCE;
        private static volatile Parser<CalculatorOptions> PARSER;

        private CalculatorOptions() {
        }

        @Override // com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptionsOrBuilder
        @Deprecated
        public boolean hasMergeFields() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptionsOrBuilder
        @Deprecated
        public boolean getMergeFields() {
            return this.mergeFields_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMergeFields(boolean value) {
            this.bitField0_ |= 1;
            this.mergeFields_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMergeFields() {
            this.bitField0_ &= -2;
            this.mergeFields_ = false;
        }

        public static CalculatorOptions parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static CalculatorOptions parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static CalculatorOptions parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static CalculatorOptions parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static CalculatorOptions parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static CalculatorOptions parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static CalculatorOptions parseFrom(InputStream input) throws IOException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static CalculatorOptions parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static CalculatorOptions parseDelimitedFrom(InputStream input) throws IOException {
            return (CalculatorOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static CalculatorOptions parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (CalculatorOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static CalculatorOptions parseFrom(CodedInputStream input) throws IOException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static CalculatorOptions parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (CalculatorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return (Builder) DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(CalculatorOptions prototype) {
            return (Builder) DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorOptionsProto$CalculatorOptions$Builder.class */
        public static final class Builder extends GeneratedMessageLite.ExtendableBuilder<CalculatorOptions, Builder> implements CalculatorOptionsOrBuilder {
            private Builder() {
                super(CalculatorOptions.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptionsOrBuilder
            @Deprecated
            public boolean hasMergeFields() {
                return ((CalculatorOptions) this.instance).hasMergeFields();
            }

            @Override // com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptionsOrBuilder
            @Deprecated
            public boolean getMergeFields() {
                return ((CalculatorOptions) this.instance).getMergeFields();
            }

            @Deprecated
            public Builder setMergeFields(boolean value) {
                copyOnWrite();
                ((CalculatorOptions) this.instance).setMergeFields(value);
                return this;
            }

            @Deprecated
            public Builder clearMergeFields() {
                copyOnWrite();
                ((CalculatorOptions) this.instance).clearMergeFields();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new CalculatorOptions();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "mergeFields_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001��\u0001\u0001\u0001\u0001������\u0001\u0007��", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<CalculatorOptions> parser = PARSER;
                    if (parser == null) {
                        synchronized (CalculatorOptions.class) {
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
            CalculatorOptions defaultInstance = new CalculatorOptions();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(CalculatorOptions.class, defaultInstance);
        }

        public static CalculatorOptions getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        /* JADX DEBUG: Type inference failed for r0v1. Raw type applied. Possible types: com.google.protobuf.Parser<MessageType>, com.google.protobuf.Parser<com.google.mediapipe.proto.CalculatorOptionsProto$CalculatorOptions> */
        public static Parser<CalculatorOptions> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}