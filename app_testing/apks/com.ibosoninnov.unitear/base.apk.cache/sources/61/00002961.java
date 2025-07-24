package com.google.mediapipe.proto;

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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/PacketGeneratorOptionsProto.class */
public final class PacketGeneratorOptionsProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/PacketGeneratorOptionsProto$PacketGeneratorConfigOrBuilder.class */
    public interface PacketGeneratorConfigOrBuilder extends MessageLiteOrBuilder {
        boolean hasPacketGenerator();

        String getPacketGenerator();

        ByteString getPacketGeneratorBytes();

        List<String> getInputSidePacketList();

        int getInputSidePacketCount();

        String getInputSidePacket(int index);

        ByteString getInputSidePacketBytes(int index);

        List<String> getExternalInputList();

        int getExternalInputCount();

        String getExternalInput(int index);

        ByteString getExternalInputBytes(int index);

        List<String> getOutputSidePacketList();

        int getOutputSidePacketCount();

        String getOutputSidePacket(int index);

        ByteString getOutputSidePacketBytes(int index);

        List<String> getExternalOutputList();

        int getExternalOutputCount();

        String getExternalOutput(int index);

        ByteString getExternalOutputBytes(int index);

        boolean hasOptions();

        PacketGeneratorOptions getOptions();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/PacketGeneratorOptionsProto$PacketGeneratorOptionsOrBuilder.class */
    public interface PacketGeneratorOptionsOrBuilder extends GeneratedMessageLite.ExtendableMessageOrBuilder<PacketGeneratorOptions, PacketGeneratorOptions.Builder> {
    }

    private PacketGeneratorOptionsProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/PacketGeneratorOptionsProto$PacketGeneratorOptions.class */
    public static final class PacketGeneratorOptions extends GeneratedMessageLite.ExtendableMessage<PacketGeneratorOptions, Builder> implements PacketGeneratorOptionsOrBuilder {
        private byte memoizedIsInitialized = 2;
        private static final PacketGeneratorOptions DEFAULT_INSTANCE;
        private static volatile Parser<PacketGeneratorOptions> PARSER;

        private PacketGeneratorOptions() {
        }

        public static PacketGeneratorOptions parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static PacketGeneratorOptions parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static PacketGeneratorOptions parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static PacketGeneratorOptions parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static PacketGeneratorOptions parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static PacketGeneratorOptions parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static PacketGeneratorOptions parseFrom(InputStream input) throws IOException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static PacketGeneratorOptions parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static PacketGeneratorOptions parseDelimitedFrom(InputStream input) throws IOException {
            return (PacketGeneratorOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static PacketGeneratorOptions parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (PacketGeneratorOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static PacketGeneratorOptions parseFrom(CodedInputStream input) throws IOException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static PacketGeneratorOptions parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (PacketGeneratorOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return (Builder) DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(PacketGeneratorOptions prototype) {
            return (Builder) DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/PacketGeneratorOptionsProto$PacketGeneratorOptions$Builder.class */
        public static final class Builder extends GeneratedMessageLite.ExtendableBuilder<PacketGeneratorOptions, Builder> implements PacketGeneratorOptionsOrBuilder {
            private Builder() {
                super(PacketGeneratorOptions.DEFAULT_INSTANCE);
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new PacketGeneratorOptions();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001��", null);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<PacketGeneratorOptions> parser = PARSER;
                    if (parser == null) {
                        synchronized (PacketGeneratorOptions.class) {
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
            PacketGeneratorOptions defaultInstance = new PacketGeneratorOptions();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(PacketGeneratorOptions.class, defaultInstance);
        }

        public static PacketGeneratorOptions getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        /* JADX DEBUG: Type inference failed for r0v1. Raw type applied. Possible types: com.google.protobuf.Parser<MessageType>, com.google.protobuf.Parser<com.google.mediapipe.proto.PacketGeneratorOptionsProto$PacketGeneratorOptions> */
        public static Parser<PacketGeneratorOptions> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/PacketGeneratorOptionsProto$PacketGeneratorConfig.class */
    public static final class PacketGeneratorConfig extends GeneratedMessageLite<PacketGeneratorConfig, Builder> implements PacketGeneratorConfigOrBuilder {
        private int bitField0_;
        public static final int PACKET_GENERATOR_FIELD_NUMBER = 1;
        public static final int INPUT_SIDE_PACKET_FIELD_NUMBER = 2;
        public static final int EXTERNAL_INPUT_FIELD_NUMBER = 1002;
        public static final int OUTPUT_SIDE_PACKET_FIELD_NUMBER = 3;
        public static final int EXTERNAL_OUTPUT_FIELD_NUMBER = 1003;
        public static final int OPTIONS_FIELD_NUMBER = 4;
        private PacketGeneratorOptions options_;
        private static final PacketGeneratorConfig DEFAULT_INSTANCE;
        private static volatile Parser<PacketGeneratorConfig> PARSER;
        private byte memoizedIsInitialized = 2;
        private String packetGenerator_ = "";
        private Internal.ProtobufList<String> inputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<String> externalInput_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<String> outputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<String> externalOutput_ = GeneratedMessageLite.emptyProtobufList();

        private PacketGeneratorConfig() {
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public boolean hasPacketGenerator() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public String getPacketGenerator() {
            return this.packetGenerator_;
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public ByteString getPacketGeneratorBytes() {
            return ByteString.copyFromUtf8(this.packetGenerator_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPacketGenerator(String value) {
            value.getClass();
            this.bitField0_ |= 1;
            this.packetGenerator_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPacketGenerator() {
            this.bitField0_ &= -2;
            this.packetGenerator_ = getDefaultInstance().getPacketGenerator();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPacketGeneratorBytes(ByteString value) {
            this.packetGenerator_ = value.toStringUtf8();
            this.bitField0_ |= 1;
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public List<String> getInputSidePacketList() {
            return this.inputSidePacket_;
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public int getInputSidePacketCount() {
            return this.inputSidePacket_.size();
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public String getInputSidePacket(int index) {
            return this.inputSidePacket_.get(index);
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
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

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public List<String> getExternalInputList() {
            return this.externalInput_;
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public int getExternalInputCount() {
            return this.externalInput_.size();
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public String getExternalInput(int index) {
            return this.externalInput_.get(index);
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
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

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public List<String> getOutputSidePacketList() {
            return this.outputSidePacket_;
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public int getOutputSidePacketCount() {
            return this.outputSidePacket_.size();
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public String getOutputSidePacket(int index) {
            return this.outputSidePacket_.get(index);
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public ByteString getOutputSidePacketBytes(int index) {
            return ByteString.copyFromUtf8(this.outputSidePacket_.get(index));
        }

        private void ensureOutputSidePacketIsMutable() {
            if (!this.outputSidePacket_.isModifiable()) {
                this.outputSidePacket_ = GeneratedMessageLite.mutableCopy(this.outputSidePacket_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOutputSidePacket(int index, String value) {
            value.getClass();
            ensureOutputSidePacketIsMutable();
            this.outputSidePacket_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addOutputSidePacket(String value) {
            value.getClass();
            ensureOutputSidePacketIsMutable();
            this.outputSidePacket_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllOutputSidePacket(Iterable<String> values) {
            ensureOutputSidePacketIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.outputSidePacket_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOutputSidePacket() {
            this.outputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addOutputSidePacketBytes(ByteString value) {
            ensureOutputSidePacketIsMutable();
            this.outputSidePacket_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public List<String> getExternalOutputList() {
            return this.externalOutput_;
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public int getExternalOutputCount() {
            return this.externalOutput_.size();
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public String getExternalOutput(int index) {
            return this.externalOutput_.get(index);
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public ByteString getExternalOutputBytes(int index) {
            return ByteString.copyFromUtf8(this.externalOutput_.get(index));
        }

        private void ensureExternalOutputIsMutable() {
            if (!this.externalOutput_.isModifiable()) {
                this.externalOutput_ = GeneratedMessageLite.mutableCopy(this.externalOutput_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setExternalOutput(int index, String value) {
            value.getClass();
            ensureExternalOutputIsMutable();
            this.externalOutput_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addExternalOutput(String value) {
            value.getClass();
            ensureExternalOutputIsMutable();
            this.externalOutput_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllExternalOutput(Iterable<String> values) {
            ensureExternalOutputIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.externalOutput_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearExternalOutput() {
            this.externalOutput_ = GeneratedMessageLite.emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addExternalOutputBytes(ByteString value) {
            ensureExternalOutputIsMutable();
            this.externalOutput_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public boolean hasOptions() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
        public PacketGeneratorOptions getOptions() {
            return this.options_ == null ? PacketGeneratorOptions.getDefaultInstance() : this.options_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOptions(PacketGeneratorOptions value) {
            value.getClass();
            this.options_ = value;
            this.bitField0_ |= 2;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeOptions(PacketGeneratorOptions value) {
            value.getClass();
            if (this.options_ != null && this.options_ != PacketGeneratorOptions.getDefaultInstance()) {
                this.options_ = ((PacketGeneratorOptions.Builder) PacketGeneratorOptions.newBuilder(this.options_).mergeFrom((PacketGeneratorOptions.Builder) value)).buildPartial();
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

        public static PacketGeneratorConfig parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static PacketGeneratorConfig parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static PacketGeneratorConfig parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static PacketGeneratorConfig parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static PacketGeneratorConfig parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static PacketGeneratorConfig parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static PacketGeneratorConfig parseFrom(InputStream input) throws IOException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static PacketGeneratorConfig parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static PacketGeneratorConfig parseDelimitedFrom(InputStream input) throws IOException {
            return (PacketGeneratorConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static PacketGeneratorConfig parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (PacketGeneratorConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static PacketGeneratorConfig parseFrom(CodedInputStream input) throws IOException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static PacketGeneratorConfig parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (PacketGeneratorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(PacketGeneratorConfig prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/PacketGeneratorOptionsProto$PacketGeneratorConfig$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<PacketGeneratorConfig, Builder> implements PacketGeneratorConfigOrBuilder {
            private Builder() {
                super(PacketGeneratorConfig.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public boolean hasPacketGenerator() {
                return ((PacketGeneratorConfig) this.instance).hasPacketGenerator();
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public String getPacketGenerator() {
                return ((PacketGeneratorConfig) this.instance).getPacketGenerator();
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public ByteString getPacketGeneratorBytes() {
                return ((PacketGeneratorConfig) this.instance).getPacketGeneratorBytes();
            }

            public Builder setPacketGenerator(String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).setPacketGenerator(value);
                return this;
            }

            public Builder clearPacketGenerator() {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).clearPacketGenerator();
                return this;
            }

            public Builder setPacketGeneratorBytes(ByteString value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).setPacketGeneratorBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public List<String> getInputSidePacketList() {
                return Collections.unmodifiableList(((PacketGeneratorConfig) this.instance).getInputSidePacketList());
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public int getInputSidePacketCount() {
                return ((PacketGeneratorConfig) this.instance).getInputSidePacketCount();
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public String getInputSidePacket(int index) {
                return ((PacketGeneratorConfig) this.instance).getInputSidePacket(index);
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public ByteString getInputSidePacketBytes(int index) {
                return ((PacketGeneratorConfig) this.instance).getInputSidePacketBytes(index);
            }

            public Builder setInputSidePacket(int index, String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).setInputSidePacket(index, value);
                return this;
            }

            public Builder addInputSidePacket(String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addInputSidePacket(value);
                return this;
            }

            public Builder addAllInputSidePacket(Iterable<String> values) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addAllInputSidePacket(values);
                return this;
            }

            public Builder clearInputSidePacket() {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).clearInputSidePacket();
                return this;
            }

            public Builder addInputSidePacketBytes(ByteString value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addInputSidePacketBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public List<String> getExternalInputList() {
                return Collections.unmodifiableList(((PacketGeneratorConfig) this.instance).getExternalInputList());
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public int getExternalInputCount() {
                return ((PacketGeneratorConfig) this.instance).getExternalInputCount();
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public String getExternalInput(int index) {
                return ((PacketGeneratorConfig) this.instance).getExternalInput(index);
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public ByteString getExternalInputBytes(int index) {
                return ((PacketGeneratorConfig) this.instance).getExternalInputBytes(index);
            }

            public Builder setExternalInput(int index, String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).setExternalInput(index, value);
                return this;
            }

            public Builder addExternalInput(String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addExternalInput(value);
                return this;
            }

            public Builder addAllExternalInput(Iterable<String> values) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addAllExternalInput(values);
                return this;
            }

            public Builder clearExternalInput() {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).clearExternalInput();
                return this;
            }

            public Builder addExternalInputBytes(ByteString value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addExternalInputBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public List<String> getOutputSidePacketList() {
                return Collections.unmodifiableList(((PacketGeneratorConfig) this.instance).getOutputSidePacketList());
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public int getOutputSidePacketCount() {
                return ((PacketGeneratorConfig) this.instance).getOutputSidePacketCount();
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public String getOutputSidePacket(int index) {
                return ((PacketGeneratorConfig) this.instance).getOutputSidePacket(index);
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public ByteString getOutputSidePacketBytes(int index) {
                return ((PacketGeneratorConfig) this.instance).getOutputSidePacketBytes(index);
            }

            public Builder setOutputSidePacket(int index, String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).setOutputSidePacket(index, value);
                return this;
            }

            public Builder addOutputSidePacket(String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addOutputSidePacket(value);
                return this;
            }

            public Builder addAllOutputSidePacket(Iterable<String> values) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addAllOutputSidePacket(values);
                return this;
            }

            public Builder clearOutputSidePacket() {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).clearOutputSidePacket();
                return this;
            }

            public Builder addOutputSidePacketBytes(ByteString value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addOutputSidePacketBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public List<String> getExternalOutputList() {
                return Collections.unmodifiableList(((PacketGeneratorConfig) this.instance).getExternalOutputList());
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public int getExternalOutputCount() {
                return ((PacketGeneratorConfig) this.instance).getExternalOutputCount();
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public String getExternalOutput(int index) {
                return ((PacketGeneratorConfig) this.instance).getExternalOutput(index);
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public ByteString getExternalOutputBytes(int index) {
                return ((PacketGeneratorConfig) this.instance).getExternalOutputBytes(index);
            }

            public Builder setExternalOutput(int index, String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).setExternalOutput(index, value);
                return this;
            }

            public Builder addExternalOutput(String value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addExternalOutput(value);
                return this;
            }

            public Builder addAllExternalOutput(Iterable<String> values) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addAllExternalOutput(values);
                return this;
            }

            public Builder clearExternalOutput() {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).clearExternalOutput();
                return this;
            }

            public Builder addExternalOutputBytes(ByteString value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).addExternalOutputBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public boolean hasOptions() {
                return ((PacketGeneratorConfig) this.instance).hasOptions();
            }

            @Override // com.google.mediapipe.proto.PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder
            public PacketGeneratorOptions getOptions() {
                return ((PacketGeneratorConfig) this.instance).getOptions();
            }

            public Builder setOptions(PacketGeneratorOptions value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).setOptions(value);
                return this;
            }

            public Builder setOptions(PacketGeneratorOptions.Builder builderForValue) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).setOptions((PacketGeneratorOptions) builderForValue.build());
                return this;
            }

            public Builder mergeOptions(PacketGeneratorOptions value) {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).mergeOptions(value);
                return this;
            }

            public Builder clearOptions() {
                copyOnWrite();
                ((PacketGeneratorConfig) this.instance).clearOptions();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new PacketGeneratorConfig();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "packetGenerator_", "inputSidePacket_", "outputSidePacket_", "options_", "externalInput_", "externalOutput_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0006��\u0001\u0001ϫ\u0006��\u0004\u0001\u0001\b��\u0002\u001a\u0003\u001a\u0004Љ\u0001Ϫ\u001aϫ\u001a", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<PacketGeneratorConfig> parser = PARSER;
                    if (parser == null) {
                        synchronized (PacketGeneratorConfig.class) {
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
            PacketGeneratorConfig defaultInstance = new PacketGeneratorConfig();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(PacketGeneratorConfig.class, defaultInstance);
        }

        public static PacketGeneratorConfig getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<PacketGeneratorConfig> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}