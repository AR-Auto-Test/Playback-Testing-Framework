package com.google.mediapipe.proto;

import com.google.mediapipe.proto.MediaPipeOptionsProto;
import com.google.protobuf.AbstractMessageLite;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.GeneratedMessageLite;
import com.google.protobuf.Internal;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageLiteOrBuilder;
import com.google.protobuf.Parser;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.List;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StreamHandlerProto.class */
public final class StreamHandlerProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StreamHandlerProto$InputStreamHandlerConfigOrBuilder.class */
    public interface InputStreamHandlerConfigOrBuilder extends MessageLiteOrBuilder {
        boolean hasInputStreamHandler();

        String getInputStreamHandler();

        ByteString getInputStreamHandlerBytes();

        boolean hasOptions();

        MediaPipeOptionsProto.MediaPipeOptions getOptions();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StreamHandlerProto$OutputStreamHandlerConfigOrBuilder.class */
    public interface OutputStreamHandlerConfigOrBuilder extends MessageLiteOrBuilder {
        boolean hasOutputStreamHandler();

        String getOutputStreamHandler();

        ByteString getOutputStreamHandlerBytes();

        List<String> getInputSidePacketList();

        int getInputSidePacketCount();

        String getInputSidePacket(int index);

        ByteString getInputSidePacketBytes(int index);

        boolean hasOptions();

        MediaPipeOptionsProto.MediaPipeOptions getOptions();
    }

    private StreamHandlerProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StreamHandlerProto$InputStreamHandlerConfig.class */
    public static final class InputStreamHandlerConfig extends GeneratedMessageLite<InputStreamHandlerConfig, Builder> implements InputStreamHandlerConfigOrBuilder {
        private int bitField0_;
        public static final int INPUT_STREAM_HANDLER_FIELD_NUMBER = 1;
        public static final int OPTIONS_FIELD_NUMBER = 3;
        private MediaPipeOptionsProto.MediaPipeOptions options_;
        private static final InputStreamHandlerConfig DEFAULT_INSTANCE;
        private static volatile Parser<InputStreamHandlerConfig> PARSER;
        private byte memoizedIsInitialized = 2;
        private String inputStreamHandler_ = "DefaultInputStreamHandler";

        private InputStreamHandlerConfig() {
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
        public boolean hasInputStreamHandler() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
        public String getInputStreamHandler() {
            return this.inputStreamHandler_;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
        public ByteString getInputStreamHandlerBytes() {
            return ByteString.copyFromUtf8(this.inputStreamHandler_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInputStreamHandler(String value) {
            value.getClass();
            this.bitField0_ |= 1;
            this.inputStreamHandler_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInputStreamHandler() {
            this.bitField0_ &= -2;
            this.inputStreamHandler_ = getDefaultInstance().getInputStreamHandler();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInputStreamHandlerBytes(ByteString value) {
            this.inputStreamHandler_ = value.toStringUtf8();
            this.bitField0_ |= 1;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
        public boolean hasOptions() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
        public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
            return this.options_ == null ? MediaPipeOptionsProto.MediaPipeOptions.getDefaultInstance() : this.options_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
            value.getClass();
            this.options_ = value;
            this.bitField0_ |= 2;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
            value.getClass();
            if (this.options_ != null && this.options_ != MediaPipeOptionsProto.MediaPipeOptions.getDefaultInstance()) {
                this.options_ = ((MediaPipeOptionsProto.MediaPipeOptions.Builder) MediaPipeOptionsProto.MediaPipeOptions.newBuilder(this.options_).mergeFrom((MediaPipeOptionsProto.MediaPipeOptions.Builder) value)).buildPartial();
            } else {
                this.options_ = value;
            }
            this.bitField0_ |= 2;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOptions() {
            this.options_ = null;
            this.bitField0_ &= -3;
        }

        public static InputStreamHandlerConfig parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputStreamHandlerConfig parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputStreamHandlerConfig parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputStreamHandlerConfig parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputStreamHandlerConfig parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputStreamHandlerConfig parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputStreamHandlerConfig parseFrom(InputStream input) throws IOException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static InputStreamHandlerConfig parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static InputStreamHandlerConfig parseDelimitedFrom(InputStream input) throws IOException {
            return (InputStreamHandlerConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static InputStreamHandlerConfig parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputStreamHandlerConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static InputStreamHandlerConfig parseFrom(CodedInputStream input) throws IOException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static InputStreamHandlerConfig parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(InputStreamHandlerConfig prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StreamHandlerProto$InputStreamHandlerConfig$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<InputStreamHandlerConfig, Builder> implements InputStreamHandlerConfigOrBuilder {
            private Builder() {
                super(InputStreamHandlerConfig.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
            public boolean hasInputStreamHandler() {
                return ((InputStreamHandlerConfig) this.instance).hasInputStreamHandler();
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
            public String getInputStreamHandler() {
                return ((InputStreamHandlerConfig) this.instance).getInputStreamHandler();
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
            public ByteString getInputStreamHandlerBytes() {
                return ((InputStreamHandlerConfig) this.instance).getInputStreamHandlerBytes();
            }

            public Builder setInputStreamHandler(String value) {
                copyOnWrite();
                ((InputStreamHandlerConfig) this.instance).setInputStreamHandler(value);
                return this;
            }

            public Builder clearInputStreamHandler() {
                copyOnWrite();
                ((InputStreamHandlerConfig) this.instance).clearInputStreamHandler();
                return this;
            }

            public Builder setInputStreamHandlerBytes(ByteString value) {
                copyOnWrite();
                ((InputStreamHandlerConfig) this.instance).setInputStreamHandlerBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
            public boolean hasOptions() {
                return ((InputStreamHandlerConfig) this.instance).hasOptions();
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.InputStreamHandlerConfigOrBuilder
            public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
                return ((InputStreamHandlerConfig) this.instance).getOptions();
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((InputStreamHandlerConfig) this.instance).setOptions(value);
                return this;
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions.Builder builderForValue) {
                copyOnWrite();
                ((InputStreamHandlerConfig) this.instance).setOptions((MediaPipeOptionsProto.MediaPipeOptions) builderForValue.build());
                return this;
            }

            public Builder mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((InputStreamHandlerConfig) this.instance).mergeOptions(value);
                return this;
            }

            public Builder clearOptions() {
                copyOnWrite();
                ((InputStreamHandlerConfig) this.instance).clearOptions();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new InputStreamHandlerConfig();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "inputStreamHandler_", "options_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0002��\u0001\u0001\u0003\u0002����\u0001\u0001\b��\u0003Љ\u0001", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<InputStreamHandlerConfig> parser = PARSER;
                    if (parser == null) {
                        synchronized (InputStreamHandlerConfig.class) {
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
            InputStreamHandlerConfig defaultInstance = new InputStreamHandlerConfig();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(InputStreamHandlerConfig.class, defaultInstance);
        }

        public static InputStreamHandlerConfig getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<InputStreamHandlerConfig> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StreamHandlerProto$OutputStreamHandlerConfig.class */
    public static final class OutputStreamHandlerConfig extends GeneratedMessageLite<OutputStreamHandlerConfig, Builder> implements OutputStreamHandlerConfigOrBuilder {
        private int bitField0_;
        public static final int OUTPUT_STREAM_HANDLER_FIELD_NUMBER = 1;
        public static final int INPUT_SIDE_PACKET_FIELD_NUMBER = 2;
        public static final int OPTIONS_FIELD_NUMBER = 3;
        private MediaPipeOptionsProto.MediaPipeOptions options_;
        private static final OutputStreamHandlerConfig DEFAULT_INSTANCE;
        private static volatile Parser<OutputStreamHandlerConfig> PARSER;
        private byte memoizedIsInitialized = 2;
        private String outputStreamHandler_ = "InOrderOutputStreamHandler";
        private Internal.ProtobufList<String> inputSidePacket_ = GeneratedMessageLite.emptyProtobufList();

        private OutputStreamHandlerConfig() {
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public boolean hasOutputStreamHandler() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public String getOutputStreamHandler() {
            return this.outputStreamHandler_;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public ByteString getOutputStreamHandlerBytes() {
            return ByteString.copyFromUtf8(this.outputStreamHandler_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOutputStreamHandler(String value) {
            value.getClass();
            this.bitField0_ |= 1;
            this.outputStreamHandler_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOutputStreamHandler() {
            this.bitField0_ &= -2;
            this.outputStreamHandler_ = getDefaultInstance().getOutputStreamHandler();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOutputStreamHandlerBytes(ByteString value) {
            this.outputStreamHandler_ = value.toStringUtf8();
            this.bitField0_ |= 1;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public List<String> getInputSidePacketList() {
            return this.inputSidePacket_;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public int getInputSidePacketCount() {
            return this.inputSidePacket_.size();
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public String getInputSidePacket(int index) {
            return this.inputSidePacket_.get(index);
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public ByteString getInputSidePacketBytes(int index) {
            return ByteString.copyFromUtf8(this.inputSidePacket_.get(index));
        }

        private void ensureInputSidePacketIsMutable() {
            if (!this.inputSidePacket_.isModifiable()) {
                this.inputSidePacket_ = GeneratedMessageLite.mutableCopy(this.inputSidePacket_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInputSidePacket(int index, String value) {
            value.getClass();
            ensureInputSidePacketIsMutable();
            this.inputSidePacket_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInputSidePacket(String value) {
            value.getClass();
            ensureInputSidePacketIsMutable();
            this.inputSidePacket_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllInputSidePacket(Iterable<String> values) {
            ensureInputSidePacketIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.inputSidePacket_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInputSidePacket() {
            this.inputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInputSidePacketBytes(ByteString value) {
            ensureInputSidePacketIsMutable();
            this.inputSidePacket_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public boolean hasOptions() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
        public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
            return this.options_ == null ? MediaPipeOptionsProto.MediaPipeOptions.getDefaultInstance() : this.options_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
            value.getClass();
            this.options_ = value;
            this.bitField0_ |= 2;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
            value.getClass();
            if (this.options_ != null && this.options_ != MediaPipeOptionsProto.MediaPipeOptions.getDefaultInstance()) {
                this.options_ = ((MediaPipeOptionsProto.MediaPipeOptions.Builder) MediaPipeOptionsProto.MediaPipeOptions.newBuilder(this.options_).mergeFrom((MediaPipeOptionsProto.MediaPipeOptions.Builder) value)).buildPartial();
            } else {
                this.options_ = value;
            }
            this.bitField0_ |= 2;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOptions() {
            this.options_ = null;
            this.bitField0_ &= -3;
        }

        public static OutputStreamHandlerConfig parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static OutputStreamHandlerConfig parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static OutputStreamHandlerConfig parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static OutputStreamHandlerConfig parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static OutputStreamHandlerConfig parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static OutputStreamHandlerConfig parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static OutputStreamHandlerConfig parseFrom(InputStream input) throws IOException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static OutputStreamHandlerConfig parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static OutputStreamHandlerConfig parseDelimitedFrom(InputStream input) throws IOException {
            return (OutputStreamHandlerConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static OutputStreamHandlerConfig parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (OutputStreamHandlerConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static OutputStreamHandlerConfig parseFrom(CodedInputStream input) throws IOException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static OutputStreamHandlerConfig parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (OutputStreamHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(OutputStreamHandlerConfig prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StreamHandlerProto$OutputStreamHandlerConfig$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<OutputStreamHandlerConfig, Builder> implements OutputStreamHandlerConfigOrBuilder {
            private Builder() {
                super(OutputStreamHandlerConfig.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public boolean hasOutputStreamHandler() {
                return ((OutputStreamHandlerConfig) this.instance).hasOutputStreamHandler();
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public String getOutputStreamHandler() {
                return ((OutputStreamHandlerConfig) this.instance).getOutputStreamHandler();
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public ByteString getOutputStreamHandlerBytes() {
                return ((OutputStreamHandlerConfig) this.instance).getOutputStreamHandlerBytes();
            }

            public Builder setOutputStreamHandler(String value) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).setOutputStreamHandler(value);
                return this;
            }

            public Builder clearOutputStreamHandler() {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).clearOutputStreamHandler();
                return this;
            }

            public Builder setOutputStreamHandlerBytes(ByteString value) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).setOutputStreamHandlerBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public List<String> getInputSidePacketList() {
                return Collections.unmodifiableList(((OutputStreamHandlerConfig) this.instance).getInputSidePacketList());
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public int getInputSidePacketCount() {
                return ((OutputStreamHandlerConfig) this.instance).getInputSidePacketCount();
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public String getInputSidePacket(int index) {
                return ((OutputStreamHandlerConfig) this.instance).getInputSidePacket(index);
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public ByteString getInputSidePacketBytes(int index) {
                return ((OutputStreamHandlerConfig) this.instance).getInputSidePacketBytes(index);
            }

            public Builder setInputSidePacket(int index, String value) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).setInputSidePacket(index, value);
                return this;
            }

            public Builder addInputSidePacket(String value) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).addInputSidePacket(value);
                return this;
            }

            public Builder addAllInputSidePacket(Iterable<String> values) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).addAllInputSidePacket(values);
                return this;
            }

            public Builder clearInputSidePacket() {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).clearInputSidePacket();
                return this;
            }

            public Builder addInputSidePacketBytes(ByteString value) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).addInputSidePacketBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public boolean hasOptions() {
                return ((OutputStreamHandlerConfig) this.instance).hasOptions();
            }

            @Override // com.google.mediapipe.proto.StreamHandlerProto.OutputStreamHandlerConfigOrBuilder
            public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
                return ((OutputStreamHandlerConfig) this.instance).getOptions();
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).setOptions(value);
                return this;
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions.Builder builderForValue) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).setOptions((MediaPipeOptionsProto.MediaPipeOptions) builderForValue.build());
                return this;
            }

            public Builder mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).mergeOptions(value);
                return this;
            }

            public Builder clearOptions() {
                copyOnWrite();
                ((OutputStreamHandlerConfig) this.instance).clearOptions();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new OutputStreamHandlerConfig();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "outputStreamHandler_", "inputSidePacket_", "options_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0003��\u0001\u0001\u0003\u0003��\u0001\u0001\u0001\b��\u0002\u001a\u0003Љ\u0001", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<OutputStreamHandlerConfig> parser = PARSER;
                    if (parser == null) {
                        synchronized (OutputStreamHandlerConfig.class) {
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
            OutputStreamHandlerConfig defaultInstance = new OutputStreamHandlerConfig();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(OutputStreamHandlerConfig.class, defaultInstance);
        }

        public static OutputStreamHandlerConfig getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<OutputStreamHandlerConfig> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}