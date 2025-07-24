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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StatusHandlerProto.class */
public final class StatusHandlerProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StatusHandlerProto$StatusHandlerConfigOrBuilder.class */
    public interface StatusHandlerConfigOrBuilder extends MessageLiteOrBuilder {
        boolean hasStatusHandler();

        String getStatusHandler();

        ByteString getStatusHandlerBytes();

        List<String> getInputSidePacketList();

        int getInputSidePacketCount();

        String getInputSidePacket(int index);

        ByteString getInputSidePacketBytes(int index);

        List<String> getExternalInputList();

        int getExternalInputCount();

        String getExternalInput(int index);

        ByteString getExternalInputBytes(int index);

        boolean hasOptions();

        MediaPipeOptionsProto.MediaPipeOptions getOptions();
    }

    private StatusHandlerProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StatusHandlerProto$StatusHandlerConfig.class */
    public static final class StatusHandlerConfig extends GeneratedMessageLite<StatusHandlerConfig, Builder> implements StatusHandlerConfigOrBuilder {
        private int bitField0_;
        public static final int STATUS_HANDLER_FIELD_NUMBER = 1;
        public static final int INPUT_SIDE_PACKET_FIELD_NUMBER = 2;
        public static final int EXTERNAL_INPUT_FIELD_NUMBER = 1002;
        public static final int OPTIONS_FIELD_NUMBER = 3;
        private MediaPipeOptionsProto.MediaPipeOptions options_;
        private static final StatusHandlerConfig DEFAULT_INSTANCE;
        private static volatile Parser<StatusHandlerConfig> PARSER;
        private byte memoizedIsInitialized = 2;
        private String statusHandler_ = "";
        private Internal.ProtobufList<String> inputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<String> externalInput_ = GeneratedMessageLite.emptyProtobufList();

        private StatusHandlerConfig() {
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public boolean hasStatusHandler() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public String getStatusHandler() {
            return this.statusHandler_;
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public ByteString getStatusHandlerBytes() {
            return ByteString.copyFromUtf8(this.statusHandler_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setStatusHandler(String value) {
            value.getClass();
            this.bitField0_ |= 1;
            this.statusHandler_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearStatusHandler() {
            this.bitField0_ &= -2;
            this.statusHandler_ = getDefaultInstance().getStatusHandler();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setStatusHandlerBytes(ByteString value) {
            this.statusHandler_ = value.toStringUtf8();
            this.bitField0_ |= 1;
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public List<String> getInputSidePacketList() {
            return this.inputSidePacket_;
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public int getInputSidePacketCount() {
            return this.inputSidePacket_.size();
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public String getInputSidePacket(int index) {
            return this.inputSidePacket_.get(index);
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
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

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public List<String> getExternalInputList() {
            return this.externalInput_;
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public int getExternalInputCount() {
            return this.externalInput_.size();
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public String getExternalInput(int index) {
            return this.externalInput_.get(index);
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public ByteString getExternalInputBytes(int index) {
            return ByteString.copyFromUtf8(this.externalInput_.get(index));
        }

        private void ensureExternalInputIsMutable() {
            if (!this.externalInput_.isModifiable()) {
                this.externalInput_ = GeneratedMessageLite.mutableCopy(this.externalInput_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setExternalInput(int index, String value) {
            value.getClass();
            ensureExternalInputIsMutable();
            this.externalInput_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addExternalInput(String value) {
            value.getClass();
            ensureExternalInputIsMutable();
            this.externalInput_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllExternalInput(Iterable<String> values) {
            ensureExternalInputIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.externalInput_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearExternalInput() {
            this.externalInput_ = GeneratedMessageLite.emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addExternalInputBytes(ByteString value) {
            ensureExternalInputIsMutable();
            this.externalInput_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
        public boolean hasOptions() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
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

        public static StatusHandlerConfig parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static StatusHandlerConfig parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static StatusHandlerConfig parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static StatusHandlerConfig parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static StatusHandlerConfig parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static StatusHandlerConfig parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static StatusHandlerConfig parseFrom(InputStream input) throws IOException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static StatusHandlerConfig parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static StatusHandlerConfig parseDelimitedFrom(InputStream input) throws IOException {
            return (StatusHandlerConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static StatusHandlerConfig parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (StatusHandlerConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static StatusHandlerConfig parseFrom(CodedInputStream input) throws IOException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static StatusHandlerConfig parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (StatusHandlerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(StatusHandlerConfig prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/StatusHandlerProto$StatusHandlerConfig$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<StatusHandlerConfig, Builder> implements StatusHandlerConfigOrBuilder {
            private Builder() {
                super(StatusHandlerConfig.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public boolean hasStatusHandler() {
                return ((StatusHandlerConfig) this.instance).hasStatusHandler();
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public String getStatusHandler() {
                return ((StatusHandlerConfig) this.instance).getStatusHandler();
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public ByteString getStatusHandlerBytes() {
                return ((StatusHandlerConfig) this.instance).getStatusHandlerBytes();
            }

            public Builder setStatusHandler(String value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).setStatusHandler(value);
                return this;
            }

            public Builder clearStatusHandler() {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).clearStatusHandler();
                return this;
            }

            public Builder setStatusHandlerBytes(ByteString value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).setStatusHandlerBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public List<String> getInputSidePacketList() {
                return Collections.unmodifiableList(((StatusHandlerConfig) this.instance).getInputSidePacketList());
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public int getInputSidePacketCount() {
                return ((StatusHandlerConfig) this.instance).getInputSidePacketCount();
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public String getInputSidePacket(int index) {
                return ((StatusHandlerConfig) this.instance).getInputSidePacket(index);
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public ByteString getInputSidePacketBytes(int index) {
                return ((StatusHandlerConfig) this.instance).getInputSidePacketBytes(index);
            }

            public Builder setInputSidePacket(int index, String value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).setInputSidePacket(index, value);
                return this;
            }

            public Builder addInputSidePacket(String value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).addInputSidePacket(value);
                return this;
            }

            public Builder addAllInputSidePacket(Iterable<String> values) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).addAllInputSidePacket(values);
                return this;
            }

            public Builder clearInputSidePacket() {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).clearInputSidePacket();
                return this;
            }

            public Builder addInputSidePacketBytes(ByteString value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).addInputSidePacketBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public List<String> getExternalInputList() {
                return Collections.unmodifiableList(((StatusHandlerConfig) this.instance).getExternalInputList());
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public int getExternalInputCount() {
                return ((StatusHandlerConfig) this.instance).getExternalInputCount();
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public String getExternalInput(int index) {
                return ((StatusHandlerConfig) this.instance).getExternalInput(index);
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public ByteString getExternalInputBytes(int index) {
                return ((StatusHandlerConfig) this.instance).getExternalInputBytes(index);
            }

            public Builder setExternalInput(int index, String value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).setExternalInput(index, value);
                return this;
            }

            public Builder addExternalInput(String value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).addExternalInput(value);
                return this;
            }

            public Builder addAllExternalInput(Iterable<String> values) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).addAllExternalInput(values);
                return this;
            }

            public Builder clearExternalInput() {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).clearExternalInput();
                return this;
            }

            public Builder addExternalInputBytes(ByteString value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).addExternalInputBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public boolean hasOptions() {
                return ((StatusHandlerConfig) this.instance).hasOptions();
            }

            @Override // com.google.mediapipe.proto.StatusHandlerProto.StatusHandlerConfigOrBuilder
            public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
                return ((StatusHandlerConfig) this.instance).getOptions();
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).setOptions(value);
                return this;
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions.Builder builderForValue) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).setOptions((MediaPipeOptionsProto.MediaPipeOptions) builderForValue.build());
                return this;
            }

            public Builder mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).mergeOptions(value);
                return this;
            }

            public Builder clearOptions() {
                copyOnWrite();
                ((StatusHandlerConfig) this.instance).clearOptions();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new StatusHandlerConfig();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "statusHandler_", "inputSidePacket_", "options_", "externalInput_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0004��\u0001\u0001Ϫ\u0004��\u0002\u0001\u0001\b��\u0002\u001a\u0003Љ\u0001Ϫ\u001a", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<StatusHandlerConfig> parser = PARSER;
                    if (parser == null) {
                        synchronized (StatusHandlerConfig.class) {
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
            StatusHandlerConfig defaultInstance = new StatusHandlerConfig();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(StatusHandlerConfig.class, defaultInstance);
        }

        public static StatusHandlerConfig getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<StatusHandlerConfig> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}