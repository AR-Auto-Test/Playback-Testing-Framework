package com.google.mediapipe.proto;

import com.google.mediapipe.proto.CalculatorOptionsProto;
import com.google.mediapipe.proto.MediaPipeOptionsProto;
import com.google.mediapipe.proto.PacketFactoryOptionsProto;
import com.google.mediapipe.proto.PacketGeneratorOptionsProto;
import com.google.mediapipe.proto.StatusHandlerProto;
import com.google.mediapipe.proto.StreamHandlerProto;
import com.google.protobuf.AbstractMessageLite;
import com.google.protobuf.Any;
import com.google.protobuf.AnyOrBuilder;
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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto.class */
public final class CalculatorProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$CalculatorGraphConfigOrBuilder.class */
    public interface CalculatorGraphConfigOrBuilder extends MessageLiteOrBuilder {
        List<CalculatorGraphConfig.Node> getNodeList();

        CalculatorGraphConfig.Node getNode(int index);

        int getNodeCount();

        List<PacketFactoryOptionsProto.PacketFactoryConfig> getPacketFactoryList();

        PacketFactoryOptionsProto.PacketFactoryConfig getPacketFactory(int index);

        int getPacketFactoryCount();

        List<PacketGeneratorOptionsProto.PacketGeneratorConfig> getPacketGeneratorList();

        PacketGeneratorOptionsProto.PacketGeneratorConfig getPacketGenerator(int index);

        int getPacketGeneratorCount();

        int getNumThreads();

        List<StatusHandlerProto.StatusHandlerConfig> getStatusHandlerList();

        StatusHandlerProto.StatusHandlerConfig getStatusHandler(int index);

        int getStatusHandlerCount();

        List<String> getInputStreamList();

        int getInputStreamCount();

        String getInputStream(int index);

        ByteString getInputStreamBytes(int index);

        List<String> getOutputStreamList();

        int getOutputStreamCount();

        String getOutputStream(int index);

        ByteString getOutputStreamBytes(int index);

        List<String> getInputSidePacketList();

        int getInputSidePacketCount();

        String getInputSidePacket(int index);

        ByteString getInputSidePacketBytes(int index);

        List<String> getOutputSidePacketList();

        int getOutputSidePacketCount();

        String getOutputSidePacket(int index);

        ByteString getOutputSidePacketBytes(int index);

        int getMaxQueueSize();

        boolean getReportDeadlock();

        boolean hasInputStreamHandler();

        StreamHandlerProto.InputStreamHandlerConfig getInputStreamHandler();

        boolean hasOutputStreamHandler();

        StreamHandlerProto.OutputStreamHandlerConfig getOutputStreamHandler();

        List<ExecutorConfig> getExecutorList();

        ExecutorConfig getExecutor(int index);

        int getExecutorCount();

        boolean hasProfilerConfig();

        ProfilerConfig getProfilerConfig();

        String getPackage();

        ByteString getPackageBytes();

        String getType();

        ByteString getTypeBytes();

        boolean hasOptions();

        MediaPipeOptionsProto.MediaPipeOptions getOptions();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$ExecutorConfigOrBuilder.class */
    public interface ExecutorConfigOrBuilder extends MessageLiteOrBuilder {
        String getName();

        ByteString getNameBytes();

        String getType();

        ByteString getTypeBytes();

        boolean hasOptions();

        MediaPipeOptionsProto.MediaPipeOptions getOptions();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputCollectionOrBuilder.class */
    public interface InputCollectionOrBuilder extends MessageLiteOrBuilder {
        String getName();

        ByteString getNameBytes();

        List<String> getSidePacketNameList();

        int getSidePacketNameCount();

        String getSidePacketName(int index);

        ByteString getSidePacketNameBytes(int index);

        List<String> getExternalInputNameList();

        int getExternalInputNameCount();

        String getExternalInputName(int index);

        ByteString getExternalInputNameBytes(int index);

        int getInputTypeValue();

        InputCollection.InputType getInputType();

        String getFileName();

        ByteString getFileNameBytes();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputCollectionSetOrBuilder.class */
    public interface InputCollectionSetOrBuilder extends MessageLiteOrBuilder {
        List<InputCollection> getInputCollectionList();

        InputCollection getInputCollection(int index);

        int getInputCollectionCount();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputStreamInfoOrBuilder.class */
    public interface InputStreamInfoOrBuilder extends MessageLiteOrBuilder {
        String getTagIndex();

        ByteString getTagIndexBytes();

        boolean getBackEdge();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$ProfilerConfigOrBuilder.class */
    public interface ProfilerConfigOrBuilder extends MessageLiteOrBuilder {
        long getHistogramIntervalSizeUsec();

        long getNumHistogramIntervals();

        @Deprecated
        boolean getEnableInputOutputLatency();

        boolean getEnableProfiler();

        boolean getEnableStreamLatency();

        boolean getUsePacketTimestampForAddedPacket();

        long getTraceLogCapacity();

        List<Integer> getTraceEventTypesDisabledList();

        int getTraceEventTypesDisabledCount();

        int getTraceEventTypesDisabled(int index);

        String getTraceLogPath();

        ByteString getTraceLogPathBytes();

        int getTraceLogCount();

        long getTraceLogIntervalUsec();

        long getTraceLogMarginUsec();

        @Deprecated
        boolean getTraceLogDurationEvents();

        int getTraceLogIntervalCount();

        boolean getTraceLogDisabled();

        boolean getTraceEnabled();

        boolean getTraceLogInstantEvents();
    }

    private CalculatorProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$ExecutorConfig.class */
    public static final class ExecutorConfig extends GeneratedMessageLite<ExecutorConfig, Builder> implements ExecutorConfigOrBuilder {
        public static final int NAME_FIELD_NUMBER = 1;
        public static final int TYPE_FIELD_NUMBER = 2;
        public static final int OPTIONS_FIELD_NUMBER = 3;
        private MediaPipeOptionsProto.MediaPipeOptions options_;
        private static final ExecutorConfig DEFAULT_INSTANCE;
        private static volatile Parser<ExecutorConfig> PARSER;
        private byte memoizedIsInitialized = 2;
        private String name_ = "";
        private String type_ = "";

        private ExecutorConfig() {
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
        public String getName() {
            return this.name_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
        public ByteString getNameBytes() {
            return ByteString.copyFromUtf8(this.name_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setName(String value) {
            value.getClass();
            this.name_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearName() {
            this.name_ = getDefaultInstance().getName();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setNameBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            this.name_ = value.toStringUtf8();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
        public String getType() {
            return this.type_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
        public ByteString getTypeBytes() {
            return ByteString.copyFromUtf8(this.type_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setType(String value) {
            value.getClass();
            this.type_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearType() {
            this.type_ = getDefaultInstance().getType();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTypeBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            this.type_ = value.toStringUtf8();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
        public boolean hasOptions() {
            return this.options_ != null;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
        public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
            return this.options_ == null ? MediaPipeOptionsProto.MediaPipeOptions.getDefaultInstance() : this.options_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
            value.getClass();
            this.options_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
            value.getClass();
            if (this.options_ != null && this.options_ != MediaPipeOptionsProto.MediaPipeOptions.getDefaultInstance()) {
                this.options_ = ((MediaPipeOptionsProto.MediaPipeOptions.Builder) MediaPipeOptionsProto.MediaPipeOptions.newBuilder(this.options_).mergeFrom((MediaPipeOptionsProto.MediaPipeOptions.Builder) value)).buildPartial();
            } else {
                this.options_ = value;
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOptions() {
            this.options_ = null;
        }

        public static ExecutorConfig parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static ExecutorConfig parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static ExecutorConfig parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static ExecutorConfig parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static ExecutorConfig parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static ExecutorConfig parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static ExecutorConfig parseFrom(InputStream input) throws IOException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static ExecutorConfig parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static ExecutorConfig parseDelimitedFrom(InputStream input) throws IOException {
            return (ExecutorConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static ExecutorConfig parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (ExecutorConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static ExecutorConfig parseFrom(CodedInputStream input) throws IOException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static ExecutorConfig parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (ExecutorConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(ExecutorConfig prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$ExecutorConfig$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<ExecutorConfig, Builder> implements ExecutorConfigOrBuilder {
            private Builder() {
                super(ExecutorConfig.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
            public String getName() {
                return ((ExecutorConfig) this.instance).getName();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
            public ByteString getNameBytes() {
                return ((ExecutorConfig) this.instance).getNameBytes();
            }

            public Builder setName(String value) {
                copyOnWrite();
                ((ExecutorConfig) this.instance).setName(value);
                return this;
            }

            public Builder clearName() {
                copyOnWrite();
                ((ExecutorConfig) this.instance).clearName();
                return this;
            }

            public Builder setNameBytes(ByteString value) {
                copyOnWrite();
                ((ExecutorConfig) this.instance).setNameBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
            public String getType() {
                return ((ExecutorConfig) this.instance).getType();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
            public ByteString getTypeBytes() {
                return ((ExecutorConfig) this.instance).getTypeBytes();
            }

            public Builder setType(String value) {
                copyOnWrite();
                ((ExecutorConfig) this.instance).setType(value);
                return this;
            }

            public Builder clearType() {
                copyOnWrite();
                ((ExecutorConfig) this.instance).clearType();
                return this;
            }

            public Builder setTypeBytes(ByteString value) {
                copyOnWrite();
                ((ExecutorConfig) this.instance).setTypeBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
            public boolean hasOptions() {
                return ((ExecutorConfig) this.instance).hasOptions();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ExecutorConfigOrBuilder
            public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
                return ((ExecutorConfig) this.instance).getOptions();
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((ExecutorConfig) this.instance).setOptions(value);
                return this;
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions.Builder builderForValue) {
                copyOnWrite();
                ((ExecutorConfig) this.instance).setOptions((MediaPipeOptionsProto.MediaPipeOptions) builderForValue.build());
                return this;
            }

            public Builder mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((ExecutorConfig) this.instance).mergeOptions(value);
                return this;
            }

            public Builder clearOptions() {
                copyOnWrite();
                ((ExecutorConfig) this.instance).clearOptions();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new ExecutorConfig();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"name_", "type_", "options_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "��\u0003����\u0001\u0003\u0003����\u0001\u0001Ȉ\u0002Ȉ\u0003Љ", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<ExecutorConfig> parser = PARSER;
                    if (parser == null) {
                        synchronized (ExecutorConfig.class) {
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
            ExecutorConfig defaultInstance = new ExecutorConfig();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(ExecutorConfig.class, defaultInstance);
        }

        public static ExecutorConfig getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<ExecutorConfig> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputCollection.class */
    public static final class InputCollection extends GeneratedMessageLite<InputCollection, Builder> implements InputCollectionOrBuilder {
        public static final int NAME_FIELD_NUMBER = 1;
        public static final int SIDE_PACKET_NAME_FIELD_NUMBER = 2;
        public static final int EXTERNAL_INPUT_NAME_FIELD_NUMBER = 1002;
        public static final int INPUT_TYPE_FIELD_NUMBER = 3;
        private int inputType_;
        public static final int FILE_NAME_FIELD_NUMBER = 4;
        private static final InputCollection DEFAULT_INSTANCE;
        private static volatile Parser<InputCollection> PARSER;
        private String name_ = "";
        private Internal.ProtobufList<String> sidePacketName_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<String> externalInputName_ = GeneratedMessageLite.emptyProtobufList();
        private String fileName_ = "";

        private InputCollection() {
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputCollection$InputType.class */
        public enum InputType implements Internal.EnumLite {
            UNKNOWN(0),
            RECORDIO(1),
            FOREIGN_RECORDIO(2),
            FOREIGN_CSV_TEXT(3),
            INVALID_UPPER_BOUND(4),
            UNRECOGNIZED(-1);
            
            public static final int UNKNOWN_VALUE = 0;
            public static final int RECORDIO_VALUE = 1;
            public static final int FOREIGN_RECORDIO_VALUE = 2;
            public static final int FOREIGN_CSV_TEXT_VALUE = 3;
            public static final int INVALID_UPPER_BOUND_VALUE = 4;
            private static final Internal.EnumLiteMap<InputType> internalValueMap = new Internal.EnumLiteMap<InputType>() { // from class: com.google.mediapipe.proto.CalculatorProto.InputCollection.InputType.1
                /* JADX DEBUG: Method merged with bridge method */
                /* JADX WARN: Can't rename method to resolve collision */
                @Override // com.google.protobuf.Internal.EnumLiteMap
                public InputType findValueByNumber(int number) {
                    return InputType.forNumber(number);
                }
            };
            private final int value;

            @Override // com.google.protobuf.Internal.EnumLite
            public final int getNumber() {
                if (this == UNRECOGNIZED) {
                    throw new IllegalArgumentException("Can't get the number of an unknown enum value.");
                }
                return this.value;
            }

            @Deprecated
            public static InputType valueOf(int value) {
                return forNumber(value);
            }

            public static InputType forNumber(int value) {
                switch (value) {
                    case 0:
                        return UNKNOWN;
                    case 1:
                        return RECORDIO;
                    case 2:
                        return FOREIGN_RECORDIO;
                    case 3:
                        return FOREIGN_CSV_TEXT;
                    case 4:
                        return INVALID_UPPER_BOUND;
                    default:
                        return null;
                }
            }

            public static Internal.EnumLiteMap<InputType> internalGetValueMap() {
                return internalValueMap;
            }

            public static Internal.EnumVerifier internalGetVerifier() {
                return InputTypeVerifier.INSTANCE;
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputCollection$InputType$InputTypeVerifier.class */
            private static final class InputTypeVerifier implements Internal.EnumVerifier {
                static final Internal.EnumVerifier INSTANCE = new InputTypeVerifier();

                private InputTypeVerifier() {
                }

                @Override // com.google.protobuf.Internal.EnumVerifier
                public boolean isInRange(int number) {
                    return InputType.forNumber(number) != null;
                }
            }

            InputType(int value) {
                this.value = value;
            }
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public String getName() {
            return this.name_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public ByteString getNameBytes() {
            return ByteString.copyFromUtf8(this.name_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setName(String value) {
            value.getClass();
            this.name_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearName() {
            this.name_ = getDefaultInstance().getName();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setNameBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            this.name_ = value.toStringUtf8();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public List<String> getSidePacketNameList() {
            return this.sidePacketName_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public int getSidePacketNameCount() {
            return this.sidePacketName_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public String getSidePacketName(int index) {
            return this.sidePacketName_.get(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public ByteString getSidePacketNameBytes(int index) {
            return ByteString.copyFromUtf8(this.sidePacketName_.get(index));
        }

        private void ensureSidePacketNameIsMutable() {
            if (!this.sidePacketName_.isModifiable()) {
                this.sidePacketName_ = GeneratedMessageLite.mutableCopy(this.sidePacketName_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setSidePacketName(int index, String value) {
            value.getClass();
            ensureSidePacketNameIsMutable();
            this.sidePacketName_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addSidePacketName(String value) {
            value.getClass();
            ensureSidePacketNameIsMutable();
            this.sidePacketName_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllSidePacketName(Iterable<String> values) {
            ensureSidePacketNameIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.sidePacketName_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearSidePacketName() {
            this.sidePacketName_ = GeneratedMessageLite.emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addSidePacketNameBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            ensureSidePacketNameIsMutable();
            this.sidePacketName_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public List<String> getExternalInputNameList() {
            return this.externalInputName_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public int getExternalInputNameCount() {
            return this.externalInputName_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public String getExternalInputName(int index) {
            return this.externalInputName_.get(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public ByteString getExternalInputNameBytes(int index) {
            return ByteString.copyFromUtf8(this.externalInputName_.get(index));
        }

        private void ensureExternalInputNameIsMutable() {
            if (!this.externalInputName_.isModifiable()) {
                this.externalInputName_ = GeneratedMessageLite.mutableCopy(this.externalInputName_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setExternalInputName(int index, String value) {
            value.getClass();
            ensureExternalInputNameIsMutable();
            this.externalInputName_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addExternalInputName(String value) {
            value.getClass();
            ensureExternalInputNameIsMutable();
            this.externalInputName_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllExternalInputName(Iterable<String> values) {
            ensureExternalInputNameIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.externalInputName_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearExternalInputName() {
            this.externalInputName_ = GeneratedMessageLite.emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addExternalInputNameBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            ensureExternalInputNameIsMutable();
            this.externalInputName_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public int getInputTypeValue() {
            return this.inputType_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public InputType getInputType() {
            InputType result = InputType.forNumber(this.inputType_);
            return result == null ? InputType.UNRECOGNIZED : result;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInputTypeValue(int value) {
            this.inputType_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInputType(InputType value) {
            this.inputType_ = value.getNumber();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInputType() {
            this.inputType_ = 0;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public String getFileName() {
            return this.fileName_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
        public ByteString getFileNameBytes() {
            return ByteString.copyFromUtf8(this.fileName_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setFileName(String value) {
            value.getClass();
            this.fileName_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearFileName() {
            this.fileName_ = getDefaultInstance().getFileName();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setFileNameBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            this.fileName_ = value.toStringUtf8();
        }

        public static InputCollection parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputCollection parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputCollection parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputCollection parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputCollection parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputCollection parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputCollection parseFrom(InputStream input) throws IOException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static InputCollection parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static InputCollection parseDelimitedFrom(InputStream input) throws IOException {
            return (InputCollection) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static InputCollection parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputCollection) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static InputCollection parseFrom(CodedInputStream input) throws IOException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static InputCollection parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputCollection) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(InputCollection prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputCollection$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<InputCollection, Builder> implements InputCollectionOrBuilder {
            private Builder() {
                super(InputCollection.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public String getName() {
                return ((InputCollection) this.instance).getName();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public ByteString getNameBytes() {
                return ((InputCollection) this.instance).getNameBytes();
            }

            public Builder setName(String value) {
                copyOnWrite();
                ((InputCollection) this.instance).setName(value);
                return this;
            }

            public Builder clearName() {
                copyOnWrite();
                ((InputCollection) this.instance).clearName();
                return this;
            }

            public Builder setNameBytes(ByteString value) {
                copyOnWrite();
                ((InputCollection) this.instance).setNameBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public List<String> getSidePacketNameList() {
                return Collections.unmodifiableList(((InputCollection) this.instance).getSidePacketNameList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public int getSidePacketNameCount() {
                return ((InputCollection) this.instance).getSidePacketNameCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public String getSidePacketName(int index) {
                return ((InputCollection) this.instance).getSidePacketName(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public ByteString getSidePacketNameBytes(int index) {
                return ((InputCollection) this.instance).getSidePacketNameBytes(index);
            }

            public Builder setSidePacketName(int index, String value) {
                copyOnWrite();
                ((InputCollection) this.instance).setSidePacketName(index, value);
                return this;
            }

            public Builder addSidePacketName(String value) {
                copyOnWrite();
                ((InputCollection) this.instance).addSidePacketName(value);
                return this;
            }

            public Builder addAllSidePacketName(Iterable<String> values) {
                copyOnWrite();
                ((InputCollection) this.instance).addAllSidePacketName(values);
                return this;
            }

            public Builder clearSidePacketName() {
                copyOnWrite();
                ((InputCollection) this.instance).clearSidePacketName();
                return this;
            }

            public Builder addSidePacketNameBytes(ByteString value) {
                copyOnWrite();
                ((InputCollection) this.instance).addSidePacketNameBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public List<String> getExternalInputNameList() {
                return Collections.unmodifiableList(((InputCollection) this.instance).getExternalInputNameList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public int getExternalInputNameCount() {
                return ((InputCollection) this.instance).getExternalInputNameCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public String getExternalInputName(int index) {
                return ((InputCollection) this.instance).getExternalInputName(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public ByteString getExternalInputNameBytes(int index) {
                return ((InputCollection) this.instance).getExternalInputNameBytes(index);
            }

            public Builder setExternalInputName(int index, String value) {
                copyOnWrite();
                ((InputCollection) this.instance).setExternalInputName(index, value);
                return this;
            }

            public Builder addExternalInputName(String value) {
                copyOnWrite();
                ((InputCollection) this.instance).addExternalInputName(value);
                return this;
            }

            public Builder addAllExternalInputName(Iterable<String> values) {
                copyOnWrite();
                ((InputCollection) this.instance).addAllExternalInputName(values);
                return this;
            }

            public Builder clearExternalInputName() {
                copyOnWrite();
                ((InputCollection) this.instance).clearExternalInputName();
                return this;
            }

            public Builder addExternalInputNameBytes(ByteString value) {
                copyOnWrite();
                ((InputCollection) this.instance).addExternalInputNameBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public int getInputTypeValue() {
                return ((InputCollection) this.instance).getInputTypeValue();
            }

            public Builder setInputTypeValue(int value) {
                copyOnWrite();
                ((InputCollection) this.instance).setInputTypeValue(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public InputType getInputType() {
                return ((InputCollection) this.instance).getInputType();
            }

            public Builder setInputType(InputType value) {
                copyOnWrite();
                ((InputCollection) this.instance).setInputType(value);
                return this;
            }

            public Builder clearInputType() {
                copyOnWrite();
                ((InputCollection) this.instance).clearInputType();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public String getFileName() {
                return ((InputCollection) this.instance).getFileName();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionOrBuilder
            public ByteString getFileNameBytes() {
                return ((InputCollection) this.instance).getFileNameBytes();
            }

            public Builder setFileName(String value) {
                copyOnWrite();
                ((InputCollection) this.instance).setFileName(value);
                return this;
            }

            public Builder clearFileName() {
                copyOnWrite();
                ((InputCollection) this.instance).clearFileName();
                return this;
            }

            public Builder setFileNameBytes(ByteString value) {
                copyOnWrite();
                ((InputCollection) this.instance).setFileNameBytes(value);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new InputCollection();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"name_", "sidePacketName_", "inputType_", "fileName_", "externalInputName_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "��\u0005����\u0001Ϫ\u0005��\u0002��\u0001Ȉ\u0002Ț\u0003\f\u0004ȈϪȚ", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<InputCollection> parser = PARSER;
                    if (parser == null) {
                        synchronized (InputCollection.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return (byte) 1;
                case SET_MEMOIZED_IS_INITIALIZED:
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            InputCollection defaultInstance = new InputCollection();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(InputCollection.class, defaultInstance);
        }

        public static InputCollection getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<InputCollection> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputCollectionSet.class */
    public static final class InputCollectionSet extends GeneratedMessageLite<InputCollectionSet, Builder> implements InputCollectionSetOrBuilder {
        public static final int INPUT_COLLECTION_FIELD_NUMBER = 1;
        private Internal.ProtobufList<InputCollection> inputCollection_ = emptyProtobufList();
        private static final InputCollectionSet DEFAULT_INSTANCE;
        private static volatile Parser<InputCollectionSet> PARSER;

        private InputCollectionSet() {
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionSetOrBuilder
        public List<InputCollection> getInputCollectionList() {
            return this.inputCollection_;
        }

        public List<? extends InputCollectionOrBuilder> getInputCollectionOrBuilderList() {
            return this.inputCollection_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionSetOrBuilder
        public int getInputCollectionCount() {
            return this.inputCollection_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionSetOrBuilder
        public InputCollection getInputCollection(int index) {
            return this.inputCollection_.get(index);
        }

        public InputCollectionOrBuilder getInputCollectionOrBuilder(int index) {
            return this.inputCollection_.get(index);
        }

        private void ensureInputCollectionIsMutable() {
            if (!this.inputCollection_.isModifiable()) {
                this.inputCollection_ = GeneratedMessageLite.mutableCopy(this.inputCollection_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInputCollection(int index, InputCollection value) {
            value.getClass();
            ensureInputCollectionIsMutable();
            this.inputCollection_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInputCollection(InputCollection value) {
            value.getClass();
            ensureInputCollectionIsMutable();
            this.inputCollection_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInputCollection(int index, InputCollection value) {
            value.getClass();
            ensureInputCollectionIsMutable();
            this.inputCollection_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllInputCollection(Iterable<? extends InputCollection> values) {
            ensureInputCollectionIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.inputCollection_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInputCollection() {
            this.inputCollection_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeInputCollection(int index) {
            ensureInputCollectionIsMutable();
            this.inputCollection_.remove(index);
        }

        public static InputCollectionSet parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputCollectionSet parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputCollectionSet parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputCollectionSet parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputCollectionSet parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputCollectionSet parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputCollectionSet parseFrom(InputStream input) throws IOException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static InputCollectionSet parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static InputCollectionSet parseDelimitedFrom(InputStream input) throws IOException {
            return (InputCollectionSet) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static InputCollectionSet parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputCollectionSet) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static InputCollectionSet parseFrom(CodedInputStream input) throws IOException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static InputCollectionSet parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputCollectionSet) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(InputCollectionSet prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputCollectionSet$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<InputCollectionSet, Builder> implements InputCollectionSetOrBuilder {
            private Builder() {
                super(InputCollectionSet.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionSetOrBuilder
            public List<InputCollection> getInputCollectionList() {
                return Collections.unmodifiableList(((InputCollectionSet) this.instance).getInputCollectionList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionSetOrBuilder
            public int getInputCollectionCount() {
                return ((InputCollectionSet) this.instance).getInputCollectionCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputCollectionSetOrBuilder
            public InputCollection getInputCollection(int index) {
                return ((InputCollectionSet) this.instance).getInputCollection(index);
            }

            public Builder setInputCollection(int index, InputCollection value) {
                copyOnWrite();
                ((InputCollectionSet) this.instance).setInputCollection(index, value);
                return this;
            }

            public Builder setInputCollection(int index, InputCollection.Builder builderForValue) {
                copyOnWrite();
                ((InputCollectionSet) this.instance).setInputCollection(index, builderForValue.build());
                return this;
            }

            public Builder addInputCollection(InputCollection value) {
                copyOnWrite();
                ((InputCollectionSet) this.instance).addInputCollection(value);
                return this;
            }

            public Builder addInputCollection(int index, InputCollection value) {
                copyOnWrite();
                ((InputCollectionSet) this.instance).addInputCollection(index, value);
                return this;
            }

            public Builder addInputCollection(InputCollection.Builder builderForValue) {
                copyOnWrite();
                ((InputCollectionSet) this.instance).addInputCollection(builderForValue.build());
                return this;
            }

            public Builder addInputCollection(int index, InputCollection.Builder builderForValue) {
                copyOnWrite();
                ((InputCollectionSet) this.instance).addInputCollection(index, builderForValue.build());
                return this;
            }

            public Builder addAllInputCollection(Iterable<? extends InputCollection> values) {
                copyOnWrite();
                ((InputCollectionSet) this.instance).addAllInputCollection(values);
                return this;
            }

            public Builder clearInputCollection() {
                copyOnWrite();
                ((InputCollectionSet) this.instance).clearInputCollection();
                return this;
            }

            public Builder removeInputCollection(int index) {
                copyOnWrite();
                ((InputCollectionSet) this.instance).removeInputCollection(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new InputCollectionSet();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"inputCollection_", InputCollection.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "��\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<InputCollectionSet> parser = PARSER;
                    if (parser == null) {
                        synchronized (InputCollectionSet.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return (byte) 1;
                case SET_MEMOIZED_IS_INITIALIZED:
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            InputCollectionSet defaultInstance = new InputCollectionSet();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(InputCollectionSet.class, defaultInstance);
        }

        public static InputCollectionSet getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<InputCollectionSet> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputStreamInfo.class */
    public static final class InputStreamInfo extends GeneratedMessageLite<InputStreamInfo, Builder> implements InputStreamInfoOrBuilder {
        public static final int TAG_INDEX_FIELD_NUMBER = 1;
        private String tagIndex_ = "";
        public static final int BACK_EDGE_FIELD_NUMBER = 2;
        private boolean backEdge_;
        private static final InputStreamInfo DEFAULT_INSTANCE;
        private static volatile Parser<InputStreamInfo> PARSER;

        private InputStreamInfo() {
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputStreamInfoOrBuilder
        public String getTagIndex() {
            return this.tagIndex_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputStreamInfoOrBuilder
        public ByteString getTagIndexBytes() {
            return ByteString.copyFromUtf8(this.tagIndex_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTagIndex(String value) {
            value.getClass();
            this.tagIndex_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTagIndex() {
            this.tagIndex_ = getDefaultInstance().getTagIndex();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTagIndexBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            this.tagIndex_ = value.toStringUtf8();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.InputStreamInfoOrBuilder
        public boolean getBackEdge() {
            return this.backEdge_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBackEdge(boolean value) {
            this.backEdge_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBackEdge() {
            this.backEdge_ = false;
        }

        public static InputStreamInfo parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputStreamInfo parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputStreamInfo parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputStreamInfo parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputStreamInfo parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static InputStreamInfo parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static InputStreamInfo parseFrom(InputStream input) throws IOException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static InputStreamInfo parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static InputStreamInfo parseDelimitedFrom(InputStream input) throws IOException {
            return (InputStreamInfo) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static InputStreamInfo parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputStreamInfo) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static InputStreamInfo parseFrom(CodedInputStream input) throws IOException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static InputStreamInfo parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (InputStreamInfo) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(InputStreamInfo prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$InputStreamInfo$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<InputStreamInfo, Builder> implements InputStreamInfoOrBuilder {
            private Builder() {
                super(InputStreamInfo.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputStreamInfoOrBuilder
            public String getTagIndex() {
                return ((InputStreamInfo) this.instance).getTagIndex();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputStreamInfoOrBuilder
            public ByteString getTagIndexBytes() {
                return ((InputStreamInfo) this.instance).getTagIndexBytes();
            }

            public Builder setTagIndex(String value) {
                copyOnWrite();
                ((InputStreamInfo) this.instance).setTagIndex(value);
                return this;
            }

            public Builder clearTagIndex() {
                copyOnWrite();
                ((InputStreamInfo) this.instance).clearTagIndex();
                return this;
            }

            public Builder setTagIndexBytes(ByteString value) {
                copyOnWrite();
                ((InputStreamInfo) this.instance).setTagIndexBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.InputStreamInfoOrBuilder
            public boolean getBackEdge() {
                return ((InputStreamInfo) this.instance).getBackEdge();
            }

            public Builder setBackEdge(boolean value) {
                copyOnWrite();
                ((InputStreamInfo) this.instance).setBackEdge(value);
                return this;
            }

            public Builder clearBackEdge() {
                copyOnWrite();
                ((InputStreamInfo) this.instance).clearBackEdge();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new InputStreamInfo();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"tagIndex_", "backEdge_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "��\u0002����\u0001\u0002\u0002������\u0001Ȉ\u0002\u0007", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<InputStreamInfo> parser = PARSER;
                    if (parser == null) {
                        synchronized (InputStreamInfo.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return (byte) 1;
                case SET_MEMOIZED_IS_INITIALIZED:
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            InputStreamInfo defaultInstance = new InputStreamInfo();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(InputStreamInfo.class, defaultInstance);
        }

        public static InputStreamInfo getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<InputStreamInfo> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$ProfilerConfig.class */
    public static final class ProfilerConfig extends GeneratedMessageLite<ProfilerConfig, Builder> implements ProfilerConfigOrBuilder {
        public static final int HISTOGRAM_INTERVAL_SIZE_USEC_FIELD_NUMBER = 1;
        private long histogramIntervalSizeUsec_;
        public static final int NUM_HISTOGRAM_INTERVALS_FIELD_NUMBER = 2;
        private long numHistogramIntervals_;
        public static final int ENABLE_INPUT_OUTPUT_LATENCY_FIELD_NUMBER = 3;
        private boolean enableInputOutputLatency_;
        public static final int ENABLE_PROFILER_FIELD_NUMBER = 4;
        private boolean enableProfiler_;
        public static final int ENABLE_STREAM_LATENCY_FIELD_NUMBER = 5;
        private boolean enableStreamLatency_;
        public static final int USE_PACKET_TIMESTAMP_FOR_ADDED_PACKET_FIELD_NUMBER = 6;
        private boolean usePacketTimestampForAddedPacket_;
        public static final int TRACE_LOG_CAPACITY_FIELD_NUMBER = 7;
        private long traceLogCapacity_;
        public static final int TRACE_EVENT_TYPES_DISABLED_FIELD_NUMBER = 8;
        public static final int TRACE_LOG_PATH_FIELD_NUMBER = 9;
        public static final int TRACE_LOG_COUNT_FIELD_NUMBER = 10;
        private int traceLogCount_;
        public static final int TRACE_LOG_INTERVAL_USEC_FIELD_NUMBER = 11;
        private long traceLogIntervalUsec_;
        public static final int TRACE_LOG_MARGIN_USEC_FIELD_NUMBER = 12;
        private long traceLogMarginUsec_;
        public static final int TRACE_LOG_DURATION_EVENTS_FIELD_NUMBER = 13;
        private boolean traceLogDurationEvents_;
        public static final int TRACE_LOG_INTERVAL_COUNT_FIELD_NUMBER = 14;
        private int traceLogIntervalCount_;
        public static final int TRACE_LOG_DISABLED_FIELD_NUMBER = 15;
        private boolean traceLogDisabled_;
        public static final int TRACE_ENABLED_FIELD_NUMBER = 16;
        private boolean traceEnabled_;
        public static final int TRACE_LOG_INSTANT_EVENTS_FIELD_NUMBER = 17;
        private boolean traceLogInstantEvents_;
        private static final ProfilerConfig DEFAULT_INSTANCE;
        private static volatile Parser<ProfilerConfig> PARSER;
        private int traceEventTypesDisabledMemoizedSerializedSize = -1;
        private Internal.IntList traceEventTypesDisabled_ = emptyIntList();
        private String traceLogPath_ = "";

        private ProfilerConfig() {
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public long getHistogramIntervalSizeUsec() {
            return this.histogramIntervalSizeUsec_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setHistogramIntervalSizeUsec(long value) {
            this.histogramIntervalSizeUsec_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearHistogramIntervalSizeUsec() {
            this.histogramIntervalSizeUsec_ = 0L;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public long getNumHistogramIntervals() {
            return this.numHistogramIntervals_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setNumHistogramIntervals(long value) {
            this.numHistogramIntervals_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearNumHistogramIntervals() {
            this.numHistogramIntervals_ = 0L;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        @Deprecated
        public boolean getEnableInputOutputLatency() {
            return this.enableInputOutputLatency_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setEnableInputOutputLatency(boolean value) {
            this.enableInputOutputLatency_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearEnableInputOutputLatency() {
            this.enableInputOutputLatency_ = false;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public boolean getEnableProfiler() {
            return this.enableProfiler_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setEnableProfiler(boolean value) {
            this.enableProfiler_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearEnableProfiler() {
            this.enableProfiler_ = false;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public boolean getEnableStreamLatency() {
            return this.enableStreamLatency_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setEnableStreamLatency(boolean value) {
            this.enableStreamLatency_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearEnableStreamLatency() {
            this.enableStreamLatency_ = false;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public boolean getUsePacketTimestampForAddedPacket() {
            return this.usePacketTimestampForAddedPacket_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setUsePacketTimestampForAddedPacket(boolean value) {
            this.usePacketTimestampForAddedPacket_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearUsePacketTimestampForAddedPacket() {
            this.usePacketTimestampForAddedPacket_ = false;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public long getTraceLogCapacity() {
            return this.traceLogCapacity_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogCapacity(long value) {
            this.traceLogCapacity_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogCapacity() {
            this.traceLogCapacity_ = 0L;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public List<Integer> getTraceEventTypesDisabledList() {
            return this.traceEventTypesDisabled_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public int getTraceEventTypesDisabledCount() {
            return this.traceEventTypesDisabled_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public int getTraceEventTypesDisabled(int index) {
            return this.traceEventTypesDisabled_.getInt(index);
        }

        private void ensureTraceEventTypesDisabledIsMutable() {
            if (!this.traceEventTypesDisabled_.isModifiable()) {
                this.traceEventTypesDisabled_ = GeneratedMessageLite.mutableCopy(this.traceEventTypesDisabled_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceEventTypesDisabled(int index, int value) {
            ensureTraceEventTypesDisabledIsMutable();
            this.traceEventTypesDisabled_.setInt(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addTraceEventTypesDisabled(int value) {
            ensureTraceEventTypesDisabledIsMutable();
            this.traceEventTypesDisabled_.addInt(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllTraceEventTypesDisabled(Iterable<? extends Integer> values) {
            ensureTraceEventTypesDisabledIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.traceEventTypesDisabled_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceEventTypesDisabled() {
            this.traceEventTypesDisabled_ = emptyIntList();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public String getTraceLogPath() {
            return this.traceLogPath_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public ByteString getTraceLogPathBytes() {
            return ByteString.copyFromUtf8(this.traceLogPath_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogPath(String value) {
            value.getClass();
            this.traceLogPath_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogPath() {
            this.traceLogPath_ = getDefaultInstance().getTraceLogPath();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogPathBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            this.traceLogPath_ = value.toStringUtf8();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public int getTraceLogCount() {
            return this.traceLogCount_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogCount(int value) {
            this.traceLogCount_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogCount() {
            this.traceLogCount_ = 0;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public long getTraceLogIntervalUsec() {
            return this.traceLogIntervalUsec_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogIntervalUsec(long value) {
            this.traceLogIntervalUsec_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogIntervalUsec() {
            this.traceLogIntervalUsec_ = 0L;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public long getTraceLogMarginUsec() {
            return this.traceLogMarginUsec_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogMarginUsec(long value) {
            this.traceLogMarginUsec_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogMarginUsec() {
            this.traceLogMarginUsec_ = 0L;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        @Deprecated
        public boolean getTraceLogDurationEvents() {
            return this.traceLogDurationEvents_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogDurationEvents(boolean value) {
            this.traceLogDurationEvents_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogDurationEvents() {
            this.traceLogDurationEvents_ = false;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public int getTraceLogIntervalCount() {
            return this.traceLogIntervalCount_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogIntervalCount(int value) {
            this.traceLogIntervalCount_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogIntervalCount() {
            this.traceLogIntervalCount_ = 0;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public boolean getTraceLogDisabled() {
            return this.traceLogDisabled_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogDisabled(boolean value) {
            this.traceLogDisabled_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogDisabled() {
            this.traceLogDisabled_ = false;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public boolean getTraceEnabled() {
            return this.traceEnabled_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceEnabled(boolean value) {
            this.traceEnabled_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceEnabled() {
            this.traceEnabled_ = false;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
        public boolean getTraceLogInstantEvents() {
            return this.traceLogInstantEvents_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTraceLogInstantEvents(boolean value) {
            this.traceLogInstantEvents_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTraceLogInstantEvents() {
            this.traceLogInstantEvents_ = false;
        }

        public static ProfilerConfig parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static ProfilerConfig parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static ProfilerConfig parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static ProfilerConfig parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static ProfilerConfig parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static ProfilerConfig parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static ProfilerConfig parseFrom(InputStream input) throws IOException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static ProfilerConfig parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static ProfilerConfig parseDelimitedFrom(InputStream input) throws IOException {
            return (ProfilerConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static ProfilerConfig parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (ProfilerConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static ProfilerConfig parseFrom(CodedInputStream input) throws IOException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static ProfilerConfig parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (ProfilerConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(ProfilerConfig prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$ProfilerConfig$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<ProfilerConfig, Builder> implements ProfilerConfigOrBuilder {
            private Builder() {
                super(ProfilerConfig.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public long getHistogramIntervalSizeUsec() {
                return ((ProfilerConfig) this.instance).getHistogramIntervalSizeUsec();
            }

            public Builder setHistogramIntervalSizeUsec(long value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setHistogramIntervalSizeUsec(value);
                return this;
            }

            public Builder clearHistogramIntervalSizeUsec() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearHistogramIntervalSizeUsec();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public long getNumHistogramIntervals() {
                return ((ProfilerConfig) this.instance).getNumHistogramIntervals();
            }

            public Builder setNumHistogramIntervals(long value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setNumHistogramIntervals(value);
                return this;
            }

            public Builder clearNumHistogramIntervals() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearNumHistogramIntervals();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            @Deprecated
            public boolean getEnableInputOutputLatency() {
                return ((ProfilerConfig) this.instance).getEnableInputOutputLatency();
            }

            @Deprecated
            public Builder setEnableInputOutputLatency(boolean value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setEnableInputOutputLatency(value);
                return this;
            }

            @Deprecated
            public Builder clearEnableInputOutputLatency() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearEnableInputOutputLatency();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public boolean getEnableProfiler() {
                return ((ProfilerConfig) this.instance).getEnableProfiler();
            }

            public Builder setEnableProfiler(boolean value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setEnableProfiler(value);
                return this;
            }

            public Builder clearEnableProfiler() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearEnableProfiler();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public boolean getEnableStreamLatency() {
                return ((ProfilerConfig) this.instance).getEnableStreamLatency();
            }

            public Builder setEnableStreamLatency(boolean value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setEnableStreamLatency(value);
                return this;
            }

            public Builder clearEnableStreamLatency() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearEnableStreamLatency();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public boolean getUsePacketTimestampForAddedPacket() {
                return ((ProfilerConfig) this.instance).getUsePacketTimestampForAddedPacket();
            }

            public Builder setUsePacketTimestampForAddedPacket(boolean value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setUsePacketTimestampForAddedPacket(value);
                return this;
            }

            public Builder clearUsePacketTimestampForAddedPacket() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearUsePacketTimestampForAddedPacket();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public long getTraceLogCapacity() {
                return ((ProfilerConfig) this.instance).getTraceLogCapacity();
            }

            public Builder setTraceLogCapacity(long value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogCapacity(value);
                return this;
            }

            public Builder clearTraceLogCapacity() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogCapacity();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public List<Integer> getTraceEventTypesDisabledList() {
                return Collections.unmodifiableList(((ProfilerConfig) this.instance).getTraceEventTypesDisabledList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public int getTraceEventTypesDisabledCount() {
                return ((ProfilerConfig) this.instance).getTraceEventTypesDisabledCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public int getTraceEventTypesDisabled(int index) {
                return ((ProfilerConfig) this.instance).getTraceEventTypesDisabled(index);
            }

            public Builder setTraceEventTypesDisabled(int index, int value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceEventTypesDisabled(index, value);
                return this;
            }

            public Builder addTraceEventTypesDisabled(int value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).addTraceEventTypesDisabled(value);
                return this;
            }

            public Builder addAllTraceEventTypesDisabled(Iterable<? extends Integer> values) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).addAllTraceEventTypesDisabled(values);
                return this;
            }

            public Builder clearTraceEventTypesDisabled() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceEventTypesDisabled();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public String getTraceLogPath() {
                return ((ProfilerConfig) this.instance).getTraceLogPath();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public ByteString getTraceLogPathBytes() {
                return ((ProfilerConfig) this.instance).getTraceLogPathBytes();
            }

            public Builder setTraceLogPath(String value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogPath(value);
                return this;
            }

            public Builder clearTraceLogPath() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogPath();
                return this;
            }

            public Builder setTraceLogPathBytes(ByteString value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogPathBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public int getTraceLogCount() {
                return ((ProfilerConfig) this.instance).getTraceLogCount();
            }

            public Builder setTraceLogCount(int value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogCount(value);
                return this;
            }

            public Builder clearTraceLogCount() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogCount();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public long getTraceLogIntervalUsec() {
                return ((ProfilerConfig) this.instance).getTraceLogIntervalUsec();
            }

            public Builder setTraceLogIntervalUsec(long value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogIntervalUsec(value);
                return this;
            }

            public Builder clearTraceLogIntervalUsec() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogIntervalUsec();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public long getTraceLogMarginUsec() {
                return ((ProfilerConfig) this.instance).getTraceLogMarginUsec();
            }

            public Builder setTraceLogMarginUsec(long value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogMarginUsec(value);
                return this;
            }

            public Builder clearTraceLogMarginUsec() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogMarginUsec();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            @Deprecated
            public boolean getTraceLogDurationEvents() {
                return ((ProfilerConfig) this.instance).getTraceLogDurationEvents();
            }

            @Deprecated
            public Builder setTraceLogDurationEvents(boolean value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogDurationEvents(value);
                return this;
            }

            @Deprecated
            public Builder clearTraceLogDurationEvents() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogDurationEvents();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public int getTraceLogIntervalCount() {
                return ((ProfilerConfig) this.instance).getTraceLogIntervalCount();
            }

            public Builder setTraceLogIntervalCount(int value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogIntervalCount(value);
                return this;
            }

            public Builder clearTraceLogIntervalCount() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogIntervalCount();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public boolean getTraceLogDisabled() {
                return ((ProfilerConfig) this.instance).getTraceLogDisabled();
            }

            public Builder setTraceLogDisabled(boolean value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogDisabled(value);
                return this;
            }

            public Builder clearTraceLogDisabled() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogDisabled();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public boolean getTraceEnabled() {
                return ((ProfilerConfig) this.instance).getTraceEnabled();
            }

            public Builder setTraceEnabled(boolean value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceEnabled(value);
                return this;
            }

            public Builder clearTraceEnabled() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceEnabled();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.ProfilerConfigOrBuilder
            public boolean getTraceLogInstantEvents() {
                return ((ProfilerConfig) this.instance).getTraceLogInstantEvents();
            }

            public Builder setTraceLogInstantEvents(boolean value) {
                copyOnWrite();
                ((ProfilerConfig) this.instance).setTraceLogInstantEvents(value);
                return this;
            }

            public Builder clearTraceLogInstantEvents() {
                copyOnWrite();
                ((ProfilerConfig) this.instance).clearTraceLogInstantEvents();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new ProfilerConfig();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"histogramIntervalSizeUsec_", "numHistogramIntervals_", "enableInputOutputLatency_", "enableProfiler_", "enableStreamLatency_", "usePacketTimestampForAddedPacket_", "traceLogCapacity_", "traceEventTypesDisabled_", "traceLogPath_", "traceLogCount_", "traceLogIntervalUsec_", "traceLogMarginUsec_", "traceLogDurationEvents_", "traceLogIntervalCount_", "traceLogDisabled_", "traceEnabled_", "traceLogInstantEvents_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "��\u0011����\u0001\u0011\u0011��\u0001��\u0001\u0002\u0002\u0002\u0003\u0007\u0004\u0007\u0005\u0007\u0006\u0007\u0007\u0002\b'\tȈ\n\u0004\u000b\u0002\f\u0002\r\u0007\u000e\u0004\u000f\u0007\u0010\u0007\u0011\u0007", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<ProfilerConfig> parser = PARSER;
                    if (parser == null) {
                        synchronized (ProfilerConfig.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return (byte) 1;
                case SET_MEMOIZED_IS_INITIALIZED:
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            ProfilerConfig defaultInstance = new ProfilerConfig();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(ProfilerConfig.class, defaultInstance);
        }

        public static ProfilerConfig getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<ProfilerConfig> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$CalculatorGraphConfig.class */
    public static final class CalculatorGraphConfig extends GeneratedMessageLite<CalculatorGraphConfig, Builder> implements CalculatorGraphConfigOrBuilder {
        public static final int NODE_FIELD_NUMBER = 1;
        public static final int PACKET_FACTORY_FIELD_NUMBER = 6;
        public static final int PACKET_GENERATOR_FIELD_NUMBER = 7;
        public static final int NUM_THREADS_FIELD_NUMBER = 8;
        private int numThreads_;
        public static final int STATUS_HANDLER_FIELD_NUMBER = 9;
        public static final int INPUT_STREAM_FIELD_NUMBER = 10;
        public static final int OUTPUT_STREAM_FIELD_NUMBER = 15;
        public static final int INPUT_SIDE_PACKET_FIELD_NUMBER = 16;
        public static final int OUTPUT_SIDE_PACKET_FIELD_NUMBER = 17;
        public static final int MAX_QUEUE_SIZE_FIELD_NUMBER = 11;
        private int maxQueueSize_;
        public static final int REPORT_DEADLOCK_FIELD_NUMBER = 21;
        private boolean reportDeadlock_;
        public static final int INPUT_STREAM_HANDLER_FIELD_NUMBER = 12;
        private StreamHandlerProto.InputStreamHandlerConfig inputStreamHandler_;
        public static final int OUTPUT_STREAM_HANDLER_FIELD_NUMBER = 13;
        private StreamHandlerProto.OutputStreamHandlerConfig outputStreamHandler_;
        public static final int EXECUTOR_FIELD_NUMBER = 14;
        public static final int PROFILER_CONFIG_FIELD_NUMBER = 18;
        private ProfilerConfig profilerConfig_;
        public static final int PACKAGE_FIELD_NUMBER = 19;
        public static final int TYPE_FIELD_NUMBER = 20;
        public static final int OPTIONS_FIELD_NUMBER = 1001;
        private MediaPipeOptionsProto.MediaPipeOptions options_;
        private static final CalculatorGraphConfig DEFAULT_INSTANCE;
        private static volatile Parser<CalculatorGraphConfig> PARSER;
        private byte memoizedIsInitialized = 2;
        private Internal.ProtobufList<Node> node_ = emptyProtobufList();
        private Internal.ProtobufList<PacketFactoryOptionsProto.PacketFactoryConfig> packetFactory_ = emptyProtobufList();
        private Internal.ProtobufList<PacketGeneratorOptionsProto.PacketGeneratorConfig> packetGenerator_ = emptyProtobufList();
        private Internal.ProtobufList<StatusHandlerProto.StatusHandlerConfig> statusHandler_ = emptyProtobufList();
        private Internal.ProtobufList<String> inputStream_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<String> outputStream_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<String> inputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<String> outputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
        private Internal.ProtobufList<ExecutorConfig> executor_ = emptyProtobufList();
        private String package_ = "";
        private String type_ = "";

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$CalculatorGraphConfig$NodeOrBuilder.class */
        public interface NodeOrBuilder extends MessageLiteOrBuilder {
            String getName();

            ByteString getNameBytes();

            String getCalculator();

            ByteString getCalculatorBytes();

            List<String> getInputStreamList();

            int getInputStreamCount();

            String getInputStream(int index);

            ByteString getInputStreamBytes(int index);

            List<String> getOutputStreamList();

            int getOutputStreamCount();

            String getOutputStream(int index);

            ByteString getOutputStreamBytes(int index);

            List<String> getInputSidePacketList();

            int getInputSidePacketCount();

            String getInputSidePacket(int index);

            ByteString getInputSidePacketBytes(int index);

            List<String> getOutputSidePacketList();

            int getOutputSidePacketCount();

            String getOutputSidePacket(int index);

            ByteString getOutputSidePacketBytes(int index);

            boolean hasOptions();

            CalculatorOptionsProto.CalculatorOptions getOptions();

            List<Any> getNodeOptionsList();

            Any getNodeOptions(int index);

            int getNodeOptionsCount();

            int getSourceLayer();

            int getBufferSizeHint();

            boolean hasInputStreamHandler();

            StreamHandlerProto.InputStreamHandlerConfig getInputStreamHandler();

            boolean hasOutputStreamHandler();

            StreamHandlerProto.OutputStreamHandlerConfig getOutputStreamHandler();

            List<InputStreamInfo> getInputStreamInfoList();

            InputStreamInfo getInputStreamInfo(int index);

            int getInputStreamInfoCount();

            String getExecutor();

            ByteString getExecutorBytes();

            @Deprecated
            boolean hasProfilerConfig();

            @Deprecated
            ProfilerConfig getProfilerConfig();

            int getMaxInFlight();

            List<String> getExternalInputList();

            int getExternalInputCount();

            String getExternalInput(int index);

            ByteString getExternalInputBytes(int index);
        }

        private CalculatorGraphConfig() {
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$CalculatorGraphConfig$Node.class */
        public static final class Node extends GeneratedMessageLite<Node, Builder> implements NodeOrBuilder {
            public static final int NAME_FIELD_NUMBER = 1;
            public static final int CALCULATOR_FIELD_NUMBER = 2;
            public static final int INPUT_STREAM_FIELD_NUMBER = 3;
            public static final int OUTPUT_STREAM_FIELD_NUMBER = 4;
            public static final int INPUT_SIDE_PACKET_FIELD_NUMBER = 5;
            public static final int OUTPUT_SIDE_PACKET_FIELD_NUMBER = 6;
            public static final int OPTIONS_FIELD_NUMBER = 7;
            private CalculatorOptionsProto.CalculatorOptions options_;
            public static final int NODE_OPTIONS_FIELD_NUMBER = 8;
            public static final int SOURCE_LAYER_FIELD_NUMBER = 9;
            private int sourceLayer_;
            public static final int BUFFER_SIZE_HINT_FIELD_NUMBER = 10;
            private int bufferSizeHint_;
            public static final int INPUT_STREAM_HANDLER_FIELD_NUMBER = 11;
            private StreamHandlerProto.InputStreamHandlerConfig inputStreamHandler_;
            public static final int OUTPUT_STREAM_HANDLER_FIELD_NUMBER = 12;
            private StreamHandlerProto.OutputStreamHandlerConfig outputStreamHandler_;
            public static final int INPUT_STREAM_INFO_FIELD_NUMBER = 13;
            public static final int EXECUTOR_FIELD_NUMBER = 14;
            public static final int PROFILER_CONFIG_FIELD_NUMBER = 15;
            private ProfilerConfig profilerConfig_;
            public static final int MAX_IN_FLIGHT_FIELD_NUMBER = 16;
            private int maxInFlight_;
            public static final int EXTERNAL_INPUT_FIELD_NUMBER = 1005;
            private static final Node DEFAULT_INSTANCE;
            private static volatile Parser<Node> PARSER;
            private byte memoizedIsInitialized = 2;
            private String name_ = "";
            private String calculator_ = "";
            private Internal.ProtobufList<String> inputStream_ = GeneratedMessageLite.emptyProtobufList();
            private Internal.ProtobufList<String> outputStream_ = GeneratedMessageLite.emptyProtobufList();
            private Internal.ProtobufList<String> inputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
            private Internal.ProtobufList<String> outputSidePacket_ = GeneratedMessageLite.emptyProtobufList();
            private Internal.ProtobufList<Any> nodeOptions_ = emptyProtobufList();
            private Internal.ProtobufList<InputStreamInfo> inputStreamInfo_ = emptyProtobufList();
            private String executor_ = "";
            private Internal.ProtobufList<String> externalInput_ = GeneratedMessageLite.emptyProtobufList();

            private Node() {
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public String getName() {
                return this.name_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public ByteString getNameBytes() {
                return ByteString.copyFromUtf8(this.name_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setName(String value) {
                value.getClass();
                this.name_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearName() {
                this.name_ = getDefaultInstance().getName();
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setNameBytes(ByteString value) {
                checkByteStringIsUtf8(value);
                this.name_ = value.toStringUtf8();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public String getCalculator() {
                return this.calculator_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public ByteString getCalculatorBytes() {
                return ByteString.copyFromUtf8(this.calculator_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setCalculator(String value) {
                value.getClass();
                this.calculator_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearCalculator() {
                this.calculator_ = getDefaultInstance().getCalculator();
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setCalculatorBytes(ByteString value) {
                checkByteStringIsUtf8(value);
                this.calculator_ = value.toStringUtf8();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public List<String> getInputStreamList() {
                return this.inputStream_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getInputStreamCount() {
                return this.inputStream_.size();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public String getInputStream(int index) {
                return this.inputStream_.get(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public ByteString getInputStreamBytes(int index) {
                return ByteString.copyFromUtf8(this.inputStream_.get(index));
            }

            private void ensureInputStreamIsMutable() {
                if (!this.inputStream_.isModifiable()) {
                    this.inputStream_ = GeneratedMessageLite.mutableCopy(this.inputStream_);
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setInputStream(int index, String value) {
                value.getClass();
                ensureInputStreamIsMutable();
                this.inputStream_.set(index, value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addInputStream(String value) {
                value.getClass();
                ensureInputStreamIsMutable();
                this.inputStream_.add(value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addAllInputStream(Iterable<String> values) {
                ensureInputStreamIsMutable();
                AbstractMessageLite.addAll((Iterable) values, (List) this.inputStream_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearInputStream() {
                this.inputStream_ = GeneratedMessageLite.emptyProtobufList();
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addInputStreamBytes(ByteString value) {
                checkByteStringIsUtf8(value);
                ensureInputStreamIsMutable();
                this.inputStream_.add(value.toStringUtf8());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public List<String> getOutputStreamList() {
                return this.outputStream_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getOutputStreamCount() {
                return this.outputStream_.size();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public String getOutputStream(int index) {
                return this.outputStream_.get(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public ByteString getOutputStreamBytes(int index) {
                return ByteString.copyFromUtf8(this.outputStream_.get(index));
            }

            private void ensureOutputStreamIsMutable() {
                if (!this.outputStream_.isModifiable()) {
                    this.outputStream_ = GeneratedMessageLite.mutableCopy(this.outputStream_);
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setOutputStream(int index, String value) {
                value.getClass();
                ensureOutputStreamIsMutable();
                this.outputStream_.set(index, value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addOutputStream(String value) {
                value.getClass();
                ensureOutputStreamIsMutable();
                this.outputStream_.add(value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addAllOutputStream(Iterable<String> values) {
                ensureOutputStreamIsMutable();
                AbstractMessageLite.addAll((Iterable) values, (List) this.outputStream_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearOutputStream() {
                this.outputStream_ = GeneratedMessageLite.emptyProtobufList();
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addOutputStreamBytes(ByteString value) {
                checkByteStringIsUtf8(value);
                ensureOutputStreamIsMutable();
                this.outputStream_.add(value.toStringUtf8());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public List<String> getInputSidePacketList() {
                return this.inputSidePacket_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getInputSidePacketCount() {
                return this.inputSidePacket_.size();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public String getInputSidePacket(int index) {
                return this.inputSidePacket_.get(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
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
                checkByteStringIsUtf8(value);
                ensureInputSidePacketIsMutable();
                this.inputSidePacket_.add(value.toStringUtf8());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public List<String> getOutputSidePacketList() {
                return this.outputSidePacket_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getOutputSidePacketCount() {
                return this.outputSidePacket_.size();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public String getOutputSidePacket(int index) {
                return this.outputSidePacket_.get(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
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
                checkByteStringIsUtf8(value);
                ensureOutputSidePacketIsMutable();
                this.outputSidePacket_.add(value.toStringUtf8());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public boolean hasOptions() {
                return this.options_ != null;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public CalculatorOptionsProto.CalculatorOptions getOptions() {
                return this.options_ == null ? CalculatorOptionsProto.CalculatorOptions.getDefaultInstance() : this.options_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setOptions(CalculatorOptionsProto.CalculatorOptions value) {
                value.getClass();
                this.options_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void mergeOptions(CalculatorOptionsProto.CalculatorOptions value) {
                value.getClass();
                if (this.options_ != null && this.options_ != CalculatorOptionsProto.CalculatorOptions.getDefaultInstance()) {
                    this.options_ = ((CalculatorOptionsProto.CalculatorOptions.Builder) CalculatorOptionsProto.CalculatorOptions.newBuilder(this.options_).mergeFrom((CalculatorOptionsProto.CalculatorOptions.Builder) value)).buildPartial();
                } else {
                    this.options_ = value;
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearOptions() {
                this.options_ = null;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public List<Any> getNodeOptionsList() {
                return this.nodeOptions_;
            }

            public List<? extends AnyOrBuilder> getNodeOptionsOrBuilderList() {
                return this.nodeOptions_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getNodeOptionsCount() {
                return this.nodeOptions_.size();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public Any getNodeOptions(int index) {
                return this.nodeOptions_.get(index);
            }

            public AnyOrBuilder getNodeOptionsOrBuilder(int index) {
                return this.nodeOptions_.get(index);
            }

            private void ensureNodeOptionsIsMutable() {
                if (!this.nodeOptions_.isModifiable()) {
                    this.nodeOptions_ = GeneratedMessageLite.mutableCopy(this.nodeOptions_);
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setNodeOptions(int index, Any value) {
                value.getClass();
                ensureNodeOptionsIsMutable();
                this.nodeOptions_.set(index, value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addNodeOptions(Any value) {
                value.getClass();
                ensureNodeOptionsIsMutable();
                this.nodeOptions_.add(value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addNodeOptions(int index, Any value) {
                value.getClass();
                ensureNodeOptionsIsMutable();
                this.nodeOptions_.add(index, value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addAllNodeOptions(Iterable<? extends Any> values) {
                ensureNodeOptionsIsMutable();
                AbstractMessageLite.addAll((Iterable) values, (List) this.nodeOptions_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearNodeOptions() {
                this.nodeOptions_ = emptyProtobufList();
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void removeNodeOptions(int index) {
                ensureNodeOptionsIsMutable();
                this.nodeOptions_.remove(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getSourceLayer() {
                return this.sourceLayer_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setSourceLayer(int value) {
                this.sourceLayer_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearSourceLayer() {
                this.sourceLayer_ = 0;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getBufferSizeHint() {
                return this.bufferSizeHint_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setBufferSizeHint(int value) {
                this.bufferSizeHint_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearBufferSizeHint() {
                this.bufferSizeHint_ = 0;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public boolean hasInputStreamHandler() {
                return this.inputStreamHandler_ != null;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public StreamHandlerProto.InputStreamHandlerConfig getInputStreamHandler() {
                return this.inputStreamHandler_ == null ? StreamHandlerProto.InputStreamHandlerConfig.getDefaultInstance() : this.inputStreamHandler_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig value) {
                value.getClass();
                this.inputStreamHandler_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void mergeInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig value) {
                value.getClass();
                if (this.inputStreamHandler_ != null && this.inputStreamHandler_ != StreamHandlerProto.InputStreamHandlerConfig.getDefaultInstance()) {
                    this.inputStreamHandler_ = StreamHandlerProto.InputStreamHandlerConfig.newBuilder(this.inputStreamHandler_).mergeFrom((StreamHandlerProto.InputStreamHandlerConfig.Builder) value).buildPartial();
                } else {
                    this.inputStreamHandler_ = value;
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearInputStreamHandler() {
                this.inputStreamHandler_ = null;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public boolean hasOutputStreamHandler() {
                return this.outputStreamHandler_ != null;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public StreamHandlerProto.OutputStreamHandlerConfig getOutputStreamHandler() {
                return this.outputStreamHandler_ == null ? StreamHandlerProto.OutputStreamHandlerConfig.getDefaultInstance() : this.outputStreamHandler_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig value) {
                value.getClass();
                this.outputStreamHandler_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void mergeOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig value) {
                value.getClass();
                if (this.outputStreamHandler_ != null && this.outputStreamHandler_ != StreamHandlerProto.OutputStreamHandlerConfig.getDefaultInstance()) {
                    this.outputStreamHandler_ = StreamHandlerProto.OutputStreamHandlerConfig.newBuilder(this.outputStreamHandler_).mergeFrom((StreamHandlerProto.OutputStreamHandlerConfig.Builder) value).buildPartial();
                } else {
                    this.outputStreamHandler_ = value;
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearOutputStreamHandler() {
                this.outputStreamHandler_ = null;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public List<InputStreamInfo> getInputStreamInfoList() {
                return this.inputStreamInfo_;
            }

            public List<? extends InputStreamInfoOrBuilder> getInputStreamInfoOrBuilderList() {
                return this.inputStreamInfo_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getInputStreamInfoCount() {
                return this.inputStreamInfo_.size();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public InputStreamInfo getInputStreamInfo(int index) {
                return this.inputStreamInfo_.get(index);
            }

            public InputStreamInfoOrBuilder getInputStreamInfoOrBuilder(int index) {
                return this.inputStreamInfo_.get(index);
            }

            private void ensureInputStreamInfoIsMutable() {
                if (!this.inputStreamInfo_.isModifiable()) {
                    this.inputStreamInfo_ = GeneratedMessageLite.mutableCopy(this.inputStreamInfo_);
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setInputStreamInfo(int index, InputStreamInfo value) {
                value.getClass();
                ensureInputStreamInfoIsMutable();
                this.inputStreamInfo_.set(index, value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addInputStreamInfo(InputStreamInfo value) {
                value.getClass();
                ensureInputStreamInfoIsMutable();
                this.inputStreamInfo_.add(value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addInputStreamInfo(int index, InputStreamInfo value) {
                value.getClass();
                ensureInputStreamInfoIsMutable();
                this.inputStreamInfo_.add(index, value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addAllInputStreamInfo(Iterable<? extends InputStreamInfo> values) {
                ensureInputStreamInfoIsMutable();
                AbstractMessageLite.addAll((Iterable) values, (List) this.inputStreamInfo_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearInputStreamInfo() {
                this.inputStreamInfo_ = emptyProtobufList();
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void removeInputStreamInfo(int index) {
                ensureInputStreamInfoIsMutable();
                this.inputStreamInfo_.remove(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public String getExecutor() {
                return this.executor_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public ByteString getExecutorBytes() {
                return ByteString.copyFromUtf8(this.executor_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setExecutor(String value) {
                value.getClass();
                this.executor_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearExecutor() {
                this.executor_ = getDefaultInstance().getExecutor();
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setExecutorBytes(ByteString value) {
                checkByteStringIsUtf8(value);
                this.executor_ = value.toStringUtf8();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            @Deprecated
            public boolean hasProfilerConfig() {
                return this.profilerConfig_ != null;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            @Deprecated
            public ProfilerConfig getProfilerConfig() {
                return this.profilerConfig_ == null ? ProfilerConfig.getDefaultInstance() : this.profilerConfig_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setProfilerConfig(ProfilerConfig value) {
                value.getClass();
                this.profilerConfig_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void mergeProfilerConfig(ProfilerConfig value) {
                value.getClass();
                if (this.profilerConfig_ != null && this.profilerConfig_ != ProfilerConfig.getDefaultInstance()) {
                    this.profilerConfig_ = ProfilerConfig.newBuilder(this.profilerConfig_).mergeFrom((ProfilerConfig.Builder) value).buildPartial();
                } else {
                    this.profilerConfig_ = value;
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearProfilerConfig() {
                this.profilerConfig_ = null;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getMaxInFlight() {
                return this.maxInFlight_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setMaxInFlight(int value) {
                this.maxInFlight_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearMaxInFlight() {
                this.maxInFlight_ = 0;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public List<String> getExternalInputList() {
                return this.externalInput_;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public int getExternalInputCount() {
                return this.externalInput_.size();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
            public String getExternalInput(int index) {
                return this.externalInput_.get(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
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
                checkByteStringIsUtf8(value);
                ensureExternalInputIsMutable();
                this.externalInput_.add(value.toStringUtf8());
            }

            public static Node parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Node parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Node parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Node parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Node parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Node parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Node parseFrom(InputStream input) throws IOException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static Node parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Node parseDelimitedFrom(InputStream input) throws IOException {
                return (Node) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static Node parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Node) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Node parseFrom(CodedInputStream input) throws IOException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static Node parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Node) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(Node prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$CalculatorGraphConfig$Node$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<Node, Builder> implements NodeOrBuilder {
                private Builder() {
                    super(Node.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public String getName() {
                    return ((Node) this.instance).getName();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public ByteString getNameBytes() {
                    return ((Node) this.instance).getNameBytes();
                }

                public Builder setName(String value) {
                    copyOnWrite();
                    ((Node) this.instance).setName(value);
                    return this;
                }

                public Builder clearName() {
                    copyOnWrite();
                    ((Node) this.instance).clearName();
                    return this;
                }

                public Builder setNameBytes(ByteString value) {
                    copyOnWrite();
                    ((Node) this.instance).setNameBytes(value);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public String getCalculator() {
                    return ((Node) this.instance).getCalculator();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public ByteString getCalculatorBytes() {
                    return ((Node) this.instance).getCalculatorBytes();
                }

                public Builder setCalculator(String value) {
                    copyOnWrite();
                    ((Node) this.instance).setCalculator(value);
                    return this;
                }

                public Builder clearCalculator() {
                    copyOnWrite();
                    ((Node) this.instance).clearCalculator();
                    return this;
                }

                public Builder setCalculatorBytes(ByteString value) {
                    copyOnWrite();
                    ((Node) this.instance).setCalculatorBytes(value);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public List<String> getInputStreamList() {
                    return Collections.unmodifiableList(((Node) this.instance).getInputStreamList());
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getInputStreamCount() {
                    return ((Node) this.instance).getInputStreamCount();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public String getInputStream(int index) {
                    return ((Node) this.instance).getInputStream(index);
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public ByteString getInputStreamBytes(int index) {
                    return ((Node) this.instance).getInputStreamBytes(index);
                }

                public Builder setInputStream(int index, String value) {
                    copyOnWrite();
                    ((Node) this.instance).setInputStream(index, value);
                    return this;
                }

                public Builder addInputStream(String value) {
                    copyOnWrite();
                    ((Node) this.instance).addInputStream(value);
                    return this;
                }

                public Builder addAllInputStream(Iterable<String> values) {
                    copyOnWrite();
                    ((Node) this.instance).addAllInputStream(values);
                    return this;
                }

                public Builder clearInputStream() {
                    copyOnWrite();
                    ((Node) this.instance).clearInputStream();
                    return this;
                }

                public Builder addInputStreamBytes(ByteString value) {
                    copyOnWrite();
                    ((Node) this.instance).addInputStreamBytes(value);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public List<String> getOutputStreamList() {
                    return Collections.unmodifiableList(((Node) this.instance).getOutputStreamList());
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getOutputStreamCount() {
                    return ((Node) this.instance).getOutputStreamCount();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public String getOutputStream(int index) {
                    return ((Node) this.instance).getOutputStream(index);
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public ByteString getOutputStreamBytes(int index) {
                    return ((Node) this.instance).getOutputStreamBytes(index);
                }

                public Builder setOutputStream(int index, String value) {
                    copyOnWrite();
                    ((Node) this.instance).setOutputStream(index, value);
                    return this;
                }

                public Builder addOutputStream(String value) {
                    copyOnWrite();
                    ((Node) this.instance).addOutputStream(value);
                    return this;
                }

                public Builder addAllOutputStream(Iterable<String> values) {
                    copyOnWrite();
                    ((Node) this.instance).addAllOutputStream(values);
                    return this;
                }

                public Builder clearOutputStream() {
                    copyOnWrite();
                    ((Node) this.instance).clearOutputStream();
                    return this;
                }

                public Builder addOutputStreamBytes(ByteString value) {
                    copyOnWrite();
                    ((Node) this.instance).addOutputStreamBytes(value);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public List<String> getInputSidePacketList() {
                    return Collections.unmodifiableList(((Node) this.instance).getInputSidePacketList());
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getInputSidePacketCount() {
                    return ((Node) this.instance).getInputSidePacketCount();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public String getInputSidePacket(int index) {
                    return ((Node) this.instance).getInputSidePacket(index);
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public ByteString getInputSidePacketBytes(int index) {
                    return ((Node) this.instance).getInputSidePacketBytes(index);
                }

                public Builder setInputSidePacket(int index, String value) {
                    copyOnWrite();
                    ((Node) this.instance).setInputSidePacket(index, value);
                    return this;
                }

                public Builder addInputSidePacket(String value) {
                    copyOnWrite();
                    ((Node) this.instance).addInputSidePacket(value);
                    return this;
                }

                public Builder addAllInputSidePacket(Iterable<String> values) {
                    copyOnWrite();
                    ((Node) this.instance).addAllInputSidePacket(values);
                    return this;
                }

                public Builder clearInputSidePacket() {
                    copyOnWrite();
                    ((Node) this.instance).clearInputSidePacket();
                    return this;
                }

                public Builder addInputSidePacketBytes(ByteString value) {
                    copyOnWrite();
                    ((Node) this.instance).addInputSidePacketBytes(value);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public List<String> getOutputSidePacketList() {
                    return Collections.unmodifiableList(((Node) this.instance).getOutputSidePacketList());
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getOutputSidePacketCount() {
                    return ((Node) this.instance).getOutputSidePacketCount();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public String getOutputSidePacket(int index) {
                    return ((Node) this.instance).getOutputSidePacket(index);
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public ByteString getOutputSidePacketBytes(int index) {
                    return ((Node) this.instance).getOutputSidePacketBytes(index);
                }

                public Builder setOutputSidePacket(int index, String value) {
                    copyOnWrite();
                    ((Node) this.instance).setOutputSidePacket(index, value);
                    return this;
                }

                public Builder addOutputSidePacket(String value) {
                    copyOnWrite();
                    ((Node) this.instance).addOutputSidePacket(value);
                    return this;
                }

                public Builder addAllOutputSidePacket(Iterable<String> values) {
                    copyOnWrite();
                    ((Node) this.instance).addAllOutputSidePacket(values);
                    return this;
                }

                public Builder clearOutputSidePacket() {
                    copyOnWrite();
                    ((Node) this.instance).clearOutputSidePacket();
                    return this;
                }

                public Builder addOutputSidePacketBytes(ByteString value) {
                    copyOnWrite();
                    ((Node) this.instance).addOutputSidePacketBytes(value);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public boolean hasOptions() {
                    return ((Node) this.instance).hasOptions();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public CalculatorOptionsProto.CalculatorOptions getOptions() {
                    return ((Node) this.instance).getOptions();
                }

                public Builder setOptions(CalculatorOptionsProto.CalculatorOptions value) {
                    copyOnWrite();
                    ((Node) this.instance).setOptions(value);
                    return this;
                }

                public Builder setOptions(CalculatorOptionsProto.CalculatorOptions.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).setOptions((CalculatorOptionsProto.CalculatorOptions) builderForValue.build());
                    return this;
                }

                public Builder mergeOptions(CalculatorOptionsProto.CalculatorOptions value) {
                    copyOnWrite();
                    ((Node) this.instance).mergeOptions(value);
                    return this;
                }

                public Builder clearOptions() {
                    copyOnWrite();
                    ((Node) this.instance).clearOptions();
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public List<Any> getNodeOptionsList() {
                    return Collections.unmodifiableList(((Node) this.instance).getNodeOptionsList());
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getNodeOptionsCount() {
                    return ((Node) this.instance).getNodeOptionsCount();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public Any getNodeOptions(int index) {
                    return ((Node) this.instance).getNodeOptions(index);
                }

                public Builder setNodeOptions(int index, Any value) {
                    copyOnWrite();
                    ((Node) this.instance).setNodeOptions(index, value);
                    return this;
                }

                public Builder setNodeOptions(int index, Any.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).setNodeOptions(index, (Any) builderForValue.build());
                    return this;
                }

                public Builder addNodeOptions(Any value) {
                    copyOnWrite();
                    ((Node) this.instance).addNodeOptions(value);
                    return this;
                }

                public Builder addNodeOptions(int index, Any value) {
                    copyOnWrite();
                    ((Node) this.instance).addNodeOptions(index, value);
                    return this;
                }

                public Builder addNodeOptions(Any.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).addNodeOptions((Any) builderForValue.build());
                    return this;
                }

                public Builder addNodeOptions(int index, Any.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).addNodeOptions(index, (Any) builderForValue.build());
                    return this;
                }

                public Builder addAllNodeOptions(Iterable<? extends Any> values) {
                    copyOnWrite();
                    ((Node) this.instance).addAllNodeOptions(values);
                    return this;
                }

                public Builder clearNodeOptions() {
                    copyOnWrite();
                    ((Node) this.instance).clearNodeOptions();
                    return this;
                }

                public Builder removeNodeOptions(int index) {
                    copyOnWrite();
                    ((Node) this.instance).removeNodeOptions(index);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getSourceLayer() {
                    return ((Node) this.instance).getSourceLayer();
                }

                public Builder setSourceLayer(int value) {
                    copyOnWrite();
                    ((Node) this.instance).setSourceLayer(value);
                    return this;
                }

                public Builder clearSourceLayer() {
                    copyOnWrite();
                    ((Node) this.instance).clearSourceLayer();
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getBufferSizeHint() {
                    return ((Node) this.instance).getBufferSizeHint();
                }

                public Builder setBufferSizeHint(int value) {
                    copyOnWrite();
                    ((Node) this.instance).setBufferSizeHint(value);
                    return this;
                }

                public Builder clearBufferSizeHint() {
                    copyOnWrite();
                    ((Node) this.instance).clearBufferSizeHint();
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public boolean hasInputStreamHandler() {
                    return ((Node) this.instance).hasInputStreamHandler();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public StreamHandlerProto.InputStreamHandlerConfig getInputStreamHandler() {
                    return ((Node) this.instance).getInputStreamHandler();
                }

                public Builder setInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig value) {
                    copyOnWrite();
                    ((Node) this.instance).setInputStreamHandler(value);
                    return this;
                }

                public Builder setInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).setInputStreamHandler(builderForValue.build());
                    return this;
                }

                public Builder mergeInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig value) {
                    copyOnWrite();
                    ((Node) this.instance).mergeInputStreamHandler(value);
                    return this;
                }

                public Builder clearInputStreamHandler() {
                    copyOnWrite();
                    ((Node) this.instance).clearInputStreamHandler();
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public boolean hasOutputStreamHandler() {
                    return ((Node) this.instance).hasOutputStreamHandler();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public StreamHandlerProto.OutputStreamHandlerConfig getOutputStreamHandler() {
                    return ((Node) this.instance).getOutputStreamHandler();
                }

                public Builder setOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig value) {
                    copyOnWrite();
                    ((Node) this.instance).setOutputStreamHandler(value);
                    return this;
                }

                public Builder setOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).setOutputStreamHandler(builderForValue.build());
                    return this;
                }

                public Builder mergeOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig value) {
                    copyOnWrite();
                    ((Node) this.instance).mergeOutputStreamHandler(value);
                    return this;
                }

                public Builder clearOutputStreamHandler() {
                    copyOnWrite();
                    ((Node) this.instance).clearOutputStreamHandler();
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public List<InputStreamInfo> getInputStreamInfoList() {
                    return Collections.unmodifiableList(((Node) this.instance).getInputStreamInfoList());
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getInputStreamInfoCount() {
                    return ((Node) this.instance).getInputStreamInfoCount();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public InputStreamInfo getInputStreamInfo(int index) {
                    return ((Node) this.instance).getInputStreamInfo(index);
                }

                public Builder setInputStreamInfo(int index, InputStreamInfo value) {
                    copyOnWrite();
                    ((Node) this.instance).setInputStreamInfo(index, value);
                    return this;
                }

                public Builder setInputStreamInfo(int index, InputStreamInfo.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).setInputStreamInfo(index, builderForValue.build());
                    return this;
                }

                public Builder addInputStreamInfo(InputStreamInfo value) {
                    copyOnWrite();
                    ((Node) this.instance).addInputStreamInfo(value);
                    return this;
                }

                public Builder addInputStreamInfo(int index, InputStreamInfo value) {
                    copyOnWrite();
                    ((Node) this.instance).addInputStreamInfo(index, value);
                    return this;
                }

                public Builder addInputStreamInfo(InputStreamInfo.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).addInputStreamInfo(builderForValue.build());
                    return this;
                }

                public Builder addInputStreamInfo(int index, InputStreamInfo.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).addInputStreamInfo(index, builderForValue.build());
                    return this;
                }

                public Builder addAllInputStreamInfo(Iterable<? extends InputStreamInfo> values) {
                    copyOnWrite();
                    ((Node) this.instance).addAllInputStreamInfo(values);
                    return this;
                }

                public Builder clearInputStreamInfo() {
                    copyOnWrite();
                    ((Node) this.instance).clearInputStreamInfo();
                    return this;
                }

                public Builder removeInputStreamInfo(int index) {
                    copyOnWrite();
                    ((Node) this.instance).removeInputStreamInfo(index);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public String getExecutor() {
                    return ((Node) this.instance).getExecutor();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public ByteString getExecutorBytes() {
                    return ((Node) this.instance).getExecutorBytes();
                }

                public Builder setExecutor(String value) {
                    copyOnWrite();
                    ((Node) this.instance).setExecutor(value);
                    return this;
                }

                public Builder clearExecutor() {
                    copyOnWrite();
                    ((Node) this.instance).clearExecutor();
                    return this;
                }

                public Builder setExecutorBytes(ByteString value) {
                    copyOnWrite();
                    ((Node) this.instance).setExecutorBytes(value);
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                @Deprecated
                public boolean hasProfilerConfig() {
                    return ((Node) this.instance).hasProfilerConfig();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                @Deprecated
                public ProfilerConfig getProfilerConfig() {
                    return ((Node) this.instance).getProfilerConfig();
                }

                @Deprecated
                public Builder setProfilerConfig(ProfilerConfig value) {
                    copyOnWrite();
                    ((Node) this.instance).setProfilerConfig(value);
                    return this;
                }

                @Deprecated
                public Builder setProfilerConfig(ProfilerConfig.Builder builderForValue) {
                    copyOnWrite();
                    ((Node) this.instance).setProfilerConfig(builderForValue.build());
                    return this;
                }

                @Deprecated
                public Builder mergeProfilerConfig(ProfilerConfig value) {
                    copyOnWrite();
                    ((Node) this.instance).mergeProfilerConfig(value);
                    return this;
                }

                @Deprecated
                public Builder clearProfilerConfig() {
                    copyOnWrite();
                    ((Node) this.instance).clearProfilerConfig();
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getMaxInFlight() {
                    return ((Node) this.instance).getMaxInFlight();
                }

                public Builder setMaxInFlight(int value) {
                    copyOnWrite();
                    ((Node) this.instance).setMaxInFlight(value);
                    return this;
                }

                public Builder clearMaxInFlight() {
                    copyOnWrite();
                    ((Node) this.instance).clearMaxInFlight();
                    return this;
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public List<String> getExternalInputList() {
                    return Collections.unmodifiableList(((Node) this.instance).getExternalInputList());
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public int getExternalInputCount() {
                    return ((Node) this.instance).getExternalInputCount();
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public String getExternalInput(int index) {
                    return ((Node) this.instance).getExternalInput(index);
                }

                @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig.NodeOrBuilder
                public ByteString getExternalInputBytes(int index) {
                    return ((Node) this.instance).getExternalInputBytes(index);
                }

                public Builder setExternalInput(int index, String value) {
                    copyOnWrite();
                    ((Node) this.instance).setExternalInput(index, value);
                    return this;
                }

                public Builder addExternalInput(String value) {
                    copyOnWrite();
                    ((Node) this.instance).addExternalInput(value);
                    return this;
                }

                public Builder addAllExternalInput(Iterable<String> values) {
                    copyOnWrite();
                    ((Node) this.instance).addAllExternalInput(values);
                    return this;
                }

                public Builder clearExternalInput() {
                    copyOnWrite();
                    ((Node) this.instance).clearExternalInput();
                    return this;
                }

                public Builder addExternalInputBytes(ByteString value) {
                    copyOnWrite();
                    ((Node) this.instance).addExternalInputBytes(value);
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new Node();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"name_", "calculator_", "inputStream_", "outputStream_", "inputSidePacket_", "outputSidePacket_", "options_", "nodeOptions_", Any.class, "sourceLayer_", "bufferSizeHint_", "inputStreamHandler_", "outputStreamHandler_", "inputStreamInfo_", InputStreamInfo.class, "executor_", "profilerConfig_", "maxInFlight_", "externalInput_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "��\u0011����\u0001ϭ\u0011��\u0007\u0003\u0001Ȉ\u0002Ȉ\u0003Ț\u0004Ț\u0005Ț\u0006Ț\u0007Љ\b\u001b\t\u0004\n\u0004\u000bЉ\fЉ\r\u001b\u000eȈ\u000f\t\u0010\u0004ϭȚ", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<Node> parser = PARSER;
                        if (parser == null) {
                            synchronized (Node.class) {
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
                Node defaultInstance = new Node();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(Node.class, defaultInstance);
            }

            public static Node getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<Node> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<Node> getNodeList() {
            return this.node_;
        }

        public List<? extends NodeOrBuilder> getNodeOrBuilderList() {
            return this.node_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getNodeCount() {
            return this.node_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public Node getNode(int index) {
            return this.node_.get(index);
        }

        public NodeOrBuilder getNodeOrBuilder(int index) {
            return this.node_.get(index);
        }

        private void ensureNodeIsMutable() {
            if (!this.node_.isModifiable()) {
                this.node_ = GeneratedMessageLite.mutableCopy(this.node_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setNode(int index, Node value) {
            value.getClass();
            ensureNodeIsMutable();
            this.node_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addNode(Node value) {
            value.getClass();
            ensureNodeIsMutable();
            this.node_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addNode(int index, Node value) {
            value.getClass();
            ensureNodeIsMutable();
            this.node_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllNode(Iterable<? extends Node> values) {
            ensureNodeIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.node_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearNode() {
            this.node_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeNode(int index) {
            ensureNodeIsMutable();
            this.node_.remove(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<PacketFactoryOptionsProto.PacketFactoryConfig> getPacketFactoryList() {
            return this.packetFactory_;
        }

        public List<? extends PacketFactoryOptionsProto.PacketFactoryConfigOrBuilder> getPacketFactoryOrBuilderList() {
            return this.packetFactory_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getPacketFactoryCount() {
            return this.packetFactory_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public PacketFactoryOptionsProto.PacketFactoryConfig getPacketFactory(int index) {
            return this.packetFactory_.get(index);
        }

        public PacketFactoryOptionsProto.PacketFactoryConfigOrBuilder getPacketFactoryOrBuilder(int index) {
            return this.packetFactory_.get(index);
        }

        private void ensurePacketFactoryIsMutable() {
            if (!this.packetFactory_.isModifiable()) {
                this.packetFactory_ = GeneratedMessageLite.mutableCopy(this.packetFactory_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPacketFactory(int index, PacketFactoryOptionsProto.PacketFactoryConfig value) {
            value.getClass();
            ensurePacketFactoryIsMutable();
            this.packetFactory_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addPacketFactory(PacketFactoryOptionsProto.PacketFactoryConfig value) {
            value.getClass();
            ensurePacketFactoryIsMutable();
            this.packetFactory_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addPacketFactory(int index, PacketFactoryOptionsProto.PacketFactoryConfig value) {
            value.getClass();
            ensurePacketFactoryIsMutable();
            this.packetFactory_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllPacketFactory(Iterable<? extends PacketFactoryOptionsProto.PacketFactoryConfig> values) {
            ensurePacketFactoryIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.packetFactory_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPacketFactory() {
            this.packetFactory_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removePacketFactory(int index) {
            ensurePacketFactoryIsMutable();
            this.packetFactory_.remove(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<PacketGeneratorOptionsProto.PacketGeneratorConfig> getPacketGeneratorList() {
            return this.packetGenerator_;
        }

        public List<? extends PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder> getPacketGeneratorOrBuilderList() {
            return this.packetGenerator_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getPacketGeneratorCount() {
            return this.packetGenerator_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public PacketGeneratorOptionsProto.PacketGeneratorConfig getPacketGenerator(int index) {
            return this.packetGenerator_.get(index);
        }

        public PacketGeneratorOptionsProto.PacketGeneratorConfigOrBuilder getPacketGeneratorOrBuilder(int index) {
            return this.packetGenerator_.get(index);
        }

        private void ensurePacketGeneratorIsMutable() {
            if (!this.packetGenerator_.isModifiable()) {
                this.packetGenerator_ = GeneratedMessageLite.mutableCopy(this.packetGenerator_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPacketGenerator(int index, PacketGeneratorOptionsProto.PacketGeneratorConfig value) {
            value.getClass();
            ensurePacketGeneratorIsMutable();
            this.packetGenerator_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addPacketGenerator(PacketGeneratorOptionsProto.PacketGeneratorConfig value) {
            value.getClass();
            ensurePacketGeneratorIsMutable();
            this.packetGenerator_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addPacketGenerator(int index, PacketGeneratorOptionsProto.PacketGeneratorConfig value) {
            value.getClass();
            ensurePacketGeneratorIsMutable();
            this.packetGenerator_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllPacketGenerator(Iterable<? extends PacketGeneratorOptionsProto.PacketGeneratorConfig> values) {
            ensurePacketGeneratorIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.packetGenerator_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPacketGenerator() {
            this.packetGenerator_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removePacketGenerator(int index) {
            ensurePacketGeneratorIsMutable();
            this.packetGenerator_.remove(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getNumThreads() {
            return this.numThreads_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setNumThreads(int value) {
            this.numThreads_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearNumThreads() {
            this.numThreads_ = 0;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<StatusHandlerProto.StatusHandlerConfig> getStatusHandlerList() {
            return this.statusHandler_;
        }

        public List<? extends StatusHandlerProto.StatusHandlerConfigOrBuilder> getStatusHandlerOrBuilderList() {
            return this.statusHandler_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getStatusHandlerCount() {
            return this.statusHandler_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public StatusHandlerProto.StatusHandlerConfig getStatusHandler(int index) {
            return this.statusHandler_.get(index);
        }

        public StatusHandlerProto.StatusHandlerConfigOrBuilder getStatusHandlerOrBuilder(int index) {
            return this.statusHandler_.get(index);
        }

        private void ensureStatusHandlerIsMutable() {
            if (!this.statusHandler_.isModifiable()) {
                this.statusHandler_ = GeneratedMessageLite.mutableCopy(this.statusHandler_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setStatusHandler(int index, StatusHandlerProto.StatusHandlerConfig value) {
            value.getClass();
            ensureStatusHandlerIsMutable();
            this.statusHandler_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addStatusHandler(StatusHandlerProto.StatusHandlerConfig value) {
            value.getClass();
            ensureStatusHandlerIsMutable();
            this.statusHandler_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addStatusHandler(int index, StatusHandlerProto.StatusHandlerConfig value) {
            value.getClass();
            ensureStatusHandlerIsMutable();
            this.statusHandler_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllStatusHandler(Iterable<? extends StatusHandlerProto.StatusHandlerConfig> values) {
            ensureStatusHandlerIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.statusHandler_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearStatusHandler() {
            this.statusHandler_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeStatusHandler(int index) {
            ensureStatusHandlerIsMutable();
            this.statusHandler_.remove(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<String> getInputStreamList() {
            return this.inputStream_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getInputStreamCount() {
            return this.inputStream_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public String getInputStream(int index) {
            return this.inputStream_.get(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public ByteString getInputStreamBytes(int index) {
            return ByteString.copyFromUtf8(this.inputStream_.get(index));
        }

        private void ensureInputStreamIsMutable() {
            if (!this.inputStream_.isModifiable()) {
                this.inputStream_ = GeneratedMessageLite.mutableCopy(this.inputStream_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInputStream(int index, String value) {
            value.getClass();
            ensureInputStreamIsMutable();
            this.inputStream_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInputStream(String value) {
            value.getClass();
            ensureInputStreamIsMutable();
            this.inputStream_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllInputStream(Iterable<String> values) {
            ensureInputStreamIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.inputStream_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInputStream() {
            this.inputStream_ = GeneratedMessageLite.emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInputStreamBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            ensureInputStreamIsMutable();
            this.inputStream_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<String> getOutputStreamList() {
            return this.outputStream_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getOutputStreamCount() {
            return this.outputStream_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public String getOutputStream(int index) {
            return this.outputStream_.get(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public ByteString getOutputStreamBytes(int index) {
            return ByteString.copyFromUtf8(this.outputStream_.get(index));
        }

        private void ensureOutputStreamIsMutable() {
            if (!this.outputStream_.isModifiable()) {
                this.outputStream_ = GeneratedMessageLite.mutableCopy(this.outputStream_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOutputStream(int index, String value) {
            value.getClass();
            ensureOutputStreamIsMutable();
            this.outputStream_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addOutputStream(String value) {
            value.getClass();
            ensureOutputStreamIsMutable();
            this.outputStream_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllOutputStream(Iterable<String> values) {
            ensureOutputStreamIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.outputStream_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOutputStream() {
            this.outputStream_ = GeneratedMessageLite.emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addOutputStreamBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            ensureOutputStreamIsMutable();
            this.outputStream_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<String> getInputSidePacketList() {
            return this.inputSidePacket_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getInputSidePacketCount() {
            return this.inputSidePacket_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public String getInputSidePacket(int index) {
            return this.inputSidePacket_.get(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
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
            checkByteStringIsUtf8(value);
            ensureInputSidePacketIsMutable();
            this.inputSidePacket_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<String> getOutputSidePacketList() {
            return this.outputSidePacket_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getOutputSidePacketCount() {
            return this.outputSidePacket_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public String getOutputSidePacket(int index) {
            return this.outputSidePacket_.get(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
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
            checkByteStringIsUtf8(value);
            ensureOutputSidePacketIsMutable();
            this.outputSidePacket_.add(value.toStringUtf8());
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getMaxQueueSize() {
            return this.maxQueueSize_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMaxQueueSize(int value) {
            this.maxQueueSize_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMaxQueueSize() {
            this.maxQueueSize_ = 0;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public boolean getReportDeadlock() {
            return this.reportDeadlock_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setReportDeadlock(boolean value) {
            this.reportDeadlock_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearReportDeadlock() {
            this.reportDeadlock_ = false;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public boolean hasInputStreamHandler() {
            return this.inputStreamHandler_ != null;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public StreamHandlerProto.InputStreamHandlerConfig getInputStreamHandler() {
            return this.inputStreamHandler_ == null ? StreamHandlerProto.InputStreamHandlerConfig.getDefaultInstance() : this.inputStreamHandler_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig value) {
            value.getClass();
            this.inputStreamHandler_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig value) {
            value.getClass();
            if (this.inputStreamHandler_ != null && this.inputStreamHandler_ != StreamHandlerProto.InputStreamHandlerConfig.getDefaultInstance()) {
                this.inputStreamHandler_ = StreamHandlerProto.InputStreamHandlerConfig.newBuilder(this.inputStreamHandler_).mergeFrom((StreamHandlerProto.InputStreamHandlerConfig.Builder) value).buildPartial();
            } else {
                this.inputStreamHandler_ = value;
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInputStreamHandler() {
            this.inputStreamHandler_ = null;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public boolean hasOutputStreamHandler() {
            return this.outputStreamHandler_ != null;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public StreamHandlerProto.OutputStreamHandlerConfig getOutputStreamHandler() {
            return this.outputStreamHandler_ == null ? StreamHandlerProto.OutputStreamHandlerConfig.getDefaultInstance() : this.outputStreamHandler_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig value) {
            value.getClass();
            this.outputStreamHandler_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig value) {
            value.getClass();
            if (this.outputStreamHandler_ != null && this.outputStreamHandler_ != StreamHandlerProto.OutputStreamHandlerConfig.getDefaultInstance()) {
                this.outputStreamHandler_ = StreamHandlerProto.OutputStreamHandlerConfig.newBuilder(this.outputStreamHandler_).mergeFrom((StreamHandlerProto.OutputStreamHandlerConfig.Builder) value).buildPartial();
            } else {
                this.outputStreamHandler_ = value;
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOutputStreamHandler() {
            this.outputStreamHandler_ = null;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public List<ExecutorConfig> getExecutorList() {
            return this.executor_;
        }

        public List<? extends ExecutorConfigOrBuilder> getExecutorOrBuilderList() {
            return this.executor_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public int getExecutorCount() {
            return this.executor_.size();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public ExecutorConfig getExecutor(int index) {
            return this.executor_.get(index);
        }

        public ExecutorConfigOrBuilder getExecutorOrBuilder(int index) {
            return this.executor_.get(index);
        }

        private void ensureExecutorIsMutable() {
            if (!this.executor_.isModifiable()) {
                this.executor_ = GeneratedMessageLite.mutableCopy(this.executor_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setExecutor(int index, ExecutorConfig value) {
            value.getClass();
            ensureExecutorIsMutable();
            this.executor_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addExecutor(ExecutorConfig value) {
            value.getClass();
            ensureExecutorIsMutable();
            this.executor_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addExecutor(int index, ExecutorConfig value) {
            value.getClass();
            ensureExecutorIsMutable();
            this.executor_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllExecutor(Iterable<? extends ExecutorConfig> values) {
            ensureExecutorIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.executor_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearExecutor() {
            this.executor_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeExecutor(int index) {
            ensureExecutorIsMutable();
            this.executor_.remove(index);
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public boolean hasProfilerConfig() {
            return this.profilerConfig_ != null;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public ProfilerConfig getProfilerConfig() {
            return this.profilerConfig_ == null ? ProfilerConfig.getDefaultInstance() : this.profilerConfig_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setProfilerConfig(ProfilerConfig value) {
            value.getClass();
            this.profilerConfig_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeProfilerConfig(ProfilerConfig value) {
            value.getClass();
            if (this.profilerConfig_ != null && this.profilerConfig_ != ProfilerConfig.getDefaultInstance()) {
                this.profilerConfig_ = ProfilerConfig.newBuilder(this.profilerConfig_).mergeFrom((ProfilerConfig.Builder) value).buildPartial();
            } else {
                this.profilerConfig_ = value;
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearProfilerConfig() {
            this.profilerConfig_ = null;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public String getPackage() {
            return this.package_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public ByteString getPackageBytes() {
            return ByteString.copyFromUtf8(this.package_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPackage(String value) {
            value.getClass();
            this.package_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPackage() {
            this.package_ = getDefaultInstance().getPackage();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPackageBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            this.package_ = value.toStringUtf8();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public String getType() {
            return this.type_;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public ByteString getTypeBytes() {
            return ByteString.copyFromUtf8(this.type_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setType(String value) {
            value.getClass();
            this.type_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearType() {
            this.type_ = getDefaultInstance().getType();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTypeBytes(ByteString value) {
            checkByteStringIsUtf8(value);
            this.type_ = value.toStringUtf8();
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public boolean hasOptions() {
            return this.options_ != null;
        }

        @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
        public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
            return this.options_ == null ? MediaPipeOptionsProto.MediaPipeOptions.getDefaultInstance() : this.options_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
            value.getClass();
            this.options_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
            value.getClass();
            if (this.options_ != null && this.options_ != MediaPipeOptionsProto.MediaPipeOptions.getDefaultInstance()) {
                this.options_ = ((MediaPipeOptionsProto.MediaPipeOptions.Builder) MediaPipeOptionsProto.MediaPipeOptions.newBuilder(this.options_).mergeFrom((MediaPipeOptionsProto.MediaPipeOptions.Builder) value)).buildPartial();
            } else {
                this.options_ = value;
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOptions() {
            this.options_ = null;
        }

        public static CalculatorGraphConfig parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static CalculatorGraphConfig parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static CalculatorGraphConfig parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static CalculatorGraphConfig parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static CalculatorGraphConfig parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static CalculatorGraphConfig parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static CalculatorGraphConfig parseFrom(InputStream input) throws IOException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static CalculatorGraphConfig parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static CalculatorGraphConfig parseDelimitedFrom(InputStream input) throws IOException {
            return (CalculatorGraphConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static CalculatorGraphConfig parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (CalculatorGraphConfig) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static CalculatorGraphConfig parseFrom(CodedInputStream input) throws IOException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static CalculatorGraphConfig parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (CalculatorGraphConfig) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(CalculatorGraphConfig prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/proto/CalculatorProto$CalculatorGraphConfig$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<CalculatorGraphConfig, Builder> implements CalculatorGraphConfigOrBuilder {
            private Builder() {
                super(CalculatorGraphConfig.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<Node> getNodeList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getNodeList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getNodeCount() {
                return ((CalculatorGraphConfig) this.instance).getNodeCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public Node getNode(int index) {
                return ((CalculatorGraphConfig) this.instance).getNode(index);
            }

            public Builder setNode(int index, Node value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setNode(index, value);
                return this;
            }

            public Builder setNode(int index, Node.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setNode(index, builderForValue.build());
                return this;
            }

            public Builder addNode(Node value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addNode(value);
                return this;
            }

            public Builder addNode(int index, Node value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addNode(index, value);
                return this;
            }

            public Builder addNode(Node.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addNode(builderForValue.build());
                return this;
            }

            public Builder addNode(int index, Node.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addNode(index, builderForValue.build());
                return this;
            }

            public Builder addAllNode(Iterable<? extends Node> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllNode(values);
                return this;
            }

            public Builder clearNode() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearNode();
                return this;
            }

            public Builder removeNode(int index) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).removeNode(index);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<PacketFactoryOptionsProto.PacketFactoryConfig> getPacketFactoryList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getPacketFactoryList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getPacketFactoryCount() {
                return ((CalculatorGraphConfig) this.instance).getPacketFactoryCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public PacketFactoryOptionsProto.PacketFactoryConfig getPacketFactory(int index) {
                return ((CalculatorGraphConfig) this.instance).getPacketFactory(index);
            }

            public Builder setPacketFactory(int index, PacketFactoryOptionsProto.PacketFactoryConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setPacketFactory(index, value);
                return this;
            }

            public Builder setPacketFactory(int index, PacketFactoryOptionsProto.PacketFactoryConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setPacketFactory(index, builderForValue.build());
                return this;
            }

            public Builder addPacketFactory(PacketFactoryOptionsProto.PacketFactoryConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addPacketFactory(value);
                return this;
            }

            public Builder addPacketFactory(int index, PacketFactoryOptionsProto.PacketFactoryConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addPacketFactory(index, value);
                return this;
            }

            public Builder addPacketFactory(PacketFactoryOptionsProto.PacketFactoryConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addPacketFactory(builderForValue.build());
                return this;
            }

            public Builder addPacketFactory(int index, PacketFactoryOptionsProto.PacketFactoryConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addPacketFactory(index, builderForValue.build());
                return this;
            }

            public Builder addAllPacketFactory(Iterable<? extends PacketFactoryOptionsProto.PacketFactoryConfig> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllPacketFactory(values);
                return this;
            }

            public Builder clearPacketFactory() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearPacketFactory();
                return this;
            }

            public Builder removePacketFactory(int index) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).removePacketFactory(index);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<PacketGeneratorOptionsProto.PacketGeneratorConfig> getPacketGeneratorList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getPacketGeneratorList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getPacketGeneratorCount() {
                return ((CalculatorGraphConfig) this.instance).getPacketGeneratorCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public PacketGeneratorOptionsProto.PacketGeneratorConfig getPacketGenerator(int index) {
                return ((CalculatorGraphConfig) this.instance).getPacketGenerator(index);
            }

            public Builder setPacketGenerator(int index, PacketGeneratorOptionsProto.PacketGeneratorConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setPacketGenerator(index, value);
                return this;
            }

            public Builder setPacketGenerator(int index, PacketGeneratorOptionsProto.PacketGeneratorConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setPacketGenerator(index, builderForValue.build());
                return this;
            }

            public Builder addPacketGenerator(PacketGeneratorOptionsProto.PacketGeneratorConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addPacketGenerator(value);
                return this;
            }

            public Builder addPacketGenerator(int index, PacketGeneratorOptionsProto.PacketGeneratorConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addPacketGenerator(index, value);
                return this;
            }

            public Builder addPacketGenerator(PacketGeneratorOptionsProto.PacketGeneratorConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addPacketGenerator(builderForValue.build());
                return this;
            }

            public Builder addPacketGenerator(int index, PacketGeneratorOptionsProto.PacketGeneratorConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addPacketGenerator(index, builderForValue.build());
                return this;
            }

            public Builder addAllPacketGenerator(Iterable<? extends PacketGeneratorOptionsProto.PacketGeneratorConfig> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllPacketGenerator(values);
                return this;
            }

            public Builder clearPacketGenerator() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearPacketGenerator();
                return this;
            }

            public Builder removePacketGenerator(int index) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).removePacketGenerator(index);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getNumThreads() {
                return ((CalculatorGraphConfig) this.instance).getNumThreads();
            }

            public Builder setNumThreads(int value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setNumThreads(value);
                return this;
            }

            public Builder clearNumThreads() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearNumThreads();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<StatusHandlerProto.StatusHandlerConfig> getStatusHandlerList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getStatusHandlerList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getStatusHandlerCount() {
                return ((CalculatorGraphConfig) this.instance).getStatusHandlerCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public StatusHandlerProto.StatusHandlerConfig getStatusHandler(int index) {
                return ((CalculatorGraphConfig) this.instance).getStatusHandler(index);
            }

            public Builder setStatusHandler(int index, StatusHandlerProto.StatusHandlerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setStatusHandler(index, value);
                return this;
            }

            public Builder setStatusHandler(int index, StatusHandlerProto.StatusHandlerConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setStatusHandler(index, builderForValue.build());
                return this;
            }

            public Builder addStatusHandler(StatusHandlerProto.StatusHandlerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addStatusHandler(value);
                return this;
            }

            public Builder addStatusHandler(int index, StatusHandlerProto.StatusHandlerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addStatusHandler(index, value);
                return this;
            }

            public Builder addStatusHandler(StatusHandlerProto.StatusHandlerConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addStatusHandler(builderForValue.build());
                return this;
            }

            public Builder addStatusHandler(int index, StatusHandlerProto.StatusHandlerConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addStatusHandler(index, builderForValue.build());
                return this;
            }

            public Builder addAllStatusHandler(Iterable<? extends StatusHandlerProto.StatusHandlerConfig> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllStatusHandler(values);
                return this;
            }

            public Builder clearStatusHandler() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearStatusHandler();
                return this;
            }

            public Builder removeStatusHandler(int index) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).removeStatusHandler(index);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<String> getInputStreamList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getInputStreamList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getInputStreamCount() {
                return ((CalculatorGraphConfig) this.instance).getInputStreamCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public String getInputStream(int index) {
                return ((CalculatorGraphConfig) this.instance).getInputStream(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public ByteString getInputStreamBytes(int index) {
                return ((CalculatorGraphConfig) this.instance).getInputStreamBytes(index);
            }

            public Builder setInputStream(int index, String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setInputStream(index, value);
                return this;
            }

            public Builder addInputStream(String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addInputStream(value);
                return this;
            }

            public Builder addAllInputStream(Iterable<String> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllInputStream(values);
                return this;
            }

            public Builder clearInputStream() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearInputStream();
                return this;
            }

            public Builder addInputStreamBytes(ByteString value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addInputStreamBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<String> getOutputStreamList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getOutputStreamList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getOutputStreamCount() {
                return ((CalculatorGraphConfig) this.instance).getOutputStreamCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public String getOutputStream(int index) {
                return ((CalculatorGraphConfig) this.instance).getOutputStream(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public ByteString getOutputStreamBytes(int index) {
                return ((CalculatorGraphConfig) this.instance).getOutputStreamBytes(index);
            }

            public Builder setOutputStream(int index, String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setOutputStream(index, value);
                return this;
            }

            public Builder addOutputStream(String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addOutputStream(value);
                return this;
            }

            public Builder addAllOutputStream(Iterable<String> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllOutputStream(values);
                return this;
            }

            public Builder clearOutputStream() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearOutputStream();
                return this;
            }

            public Builder addOutputStreamBytes(ByteString value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addOutputStreamBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<String> getInputSidePacketList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getInputSidePacketList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getInputSidePacketCount() {
                return ((CalculatorGraphConfig) this.instance).getInputSidePacketCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public String getInputSidePacket(int index) {
                return ((CalculatorGraphConfig) this.instance).getInputSidePacket(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public ByteString getInputSidePacketBytes(int index) {
                return ((CalculatorGraphConfig) this.instance).getInputSidePacketBytes(index);
            }

            public Builder setInputSidePacket(int index, String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setInputSidePacket(index, value);
                return this;
            }

            public Builder addInputSidePacket(String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addInputSidePacket(value);
                return this;
            }

            public Builder addAllInputSidePacket(Iterable<String> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllInputSidePacket(values);
                return this;
            }

            public Builder clearInputSidePacket() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearInputSidePacket();
                return this;
            }

            public Builder addInputSidePacketBytes(ByteString value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addInputSidePacketBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<String> getOutputSidePacketList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getOutputSidePacketList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getOutputSidePacketCount() {
                return ((CalculatorGraphConfig) this.instance).getOutputSidePacketCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public String getOutputSidePacket(int index) {
                return ((CalculatorGraphConfig) this.instance).getOutputSidePacket(index);
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public ByteString getOutputSidePacketBytes(int index) {
                return ((CalculatorGraphConfig) this.instance).getOutputSidePacketBytes(index);
            }

            public Builder setOutputSidePacket(int index, String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setOutputSidePacket(index, value);
                return this;
            }

            public Builder addOutputSidePacket(String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addOutputSidePacket(value);
                return this;
            }

            public Builder addAllOutputSidePacket(Iterable<String> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllOutputSidePacket(values);
                return this;
            }

            public Builder clearOutputSidePacket() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearOutputSidePacket();
                return this;
            }

            public Builder addOutputSidePacketBytes(ByteString value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addOutputSidePacketBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getMaxQueueSize() {
                return ((CalculatorGraphConfig) this.instance).getMaxQueueSize();
            }

            public Builder setMaxQueueSize(int value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setMaxQueueSize(value);
                return this;
            }

            public Builder clearMaxQueueSize() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearMaxQueueSize();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public boolean getReportDeadlock() {
                return ((CalculatorGraphConfig) this.instance).getReportDeadlock();
            }

            public Builder setReportDeadlock(boolean value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setReportDeadlock(value);
                return this;
            }

            public Builder clearReportDeadlock() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearReportDeadlock();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public boolean hasInputStreamHandler() {
                return ((CalculatorGraphConfig) this.instance).hasInputStreamHandler();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public StreamHandlerProto.InputStreamHandlerConfig getInputStreamHandler() {
                return ((CalculatorGraphConfig) this.instance).getInputStreamHandler();
            }

            public Builder setInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setInputStreamHandler(value);
                return this;
            }

            public Builder setInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setInputStreamHandler(builderForValue.build());
                return this;
            }

            public Builder mergeInputStreamHandler(StreamHandlerProto.InputStreamHandlerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).mergeInputStreamHandler(value);
                return this;
            }

            public Builder clearInputStreamHandler() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearInputStreamHandler();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public boolean hasOutputStreamHandler() {
                return ((CalculatorGraphConfig) this.instance).hasOutputStreamHandler();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public StreamHandlerProto.OutputStreamHandlerConfig getOutputStreamHandler() {
                return ((CalculatorGraphConfig) this.instance).getOutputStreamHandler();
            }

            public Builder setOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setOutputStreamHandler(value);
                return this;
            }

            public Builder setOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setOutputStreamHandler(builderForValue.build());
                return this;
            }

            public Builder mergeOutputStreamHandler(StreamHandlerProto.OutputStreamHandlerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).mergeOutputStreamHandler(value);
                return this;
            }

            public Builder clearOutputStreamHandler() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearOutputStreamHandler();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public List<ExecutorConfig> getExecutorList() {
                return Collections.unmodifiableList(((CalculatorGraphConfig) this.instance).getExecutorList());
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public int getExecutorCount() {
                return ((CalculatorGraphConfig) this.instance).getExecutorCount();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public ExecutorConfig getExecutor(int index) {
                return ((CalculatorGraphConfig) this.instance).getExecutor(index);
            }

            public Builder setExecutor(int index, ExecutorConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setExecutor(index, value);
                return this;
            }

            public Builder setExecutor(int index, ExecutorConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setExecutor(index, builderForValue.build());
                return this;
            }

            public Builder addExecutor(ExecutorConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addExecutor(value);
                return this;
            }

            public Builder addExecutor(int index, ExecutorConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addExecutor(index, value);
                return this;
            }

            public Builder addExecutor(ExecutorConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addExecutor(builderForValue.build());
                return this;
            }

            public Builder addExecutor(int index, ExecutorConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addExecutor(index, builderForValue.build());
                return this;
            }

            public Builder addAllExecutor(Iterable<? extends ExecutorConfig> values) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).addAllExecutor(values);
                return this;
            }

            public Builder clearExecutor() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearExecutor();
                return this;
            }

            public Builder removeExecutor(int index) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).removeExecutor(index);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public boolean hasProfilerConfig() {
                return ((CalculatorGraphConfig) this.instance).hasProfilerConfig();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public ProfilerConfig getProfilerConfig() {
                return ((CalculatorGraphConfig) this.instance).getProfilerConfig();
            }

            public Builder setProfilerConfig(ProfilerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setProfilerConfig(value);
                return this;
            }

            public Builder setProfilerConfig(ProfilerConfig.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setProfilerConfig(builderForValue.build());
                return this;
            }

            public Builder mergeProfilerConfig(ProfilerConfig value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).mergeProfilerConfig(value);
                return this;
            }

            public Builder clearProfilerConfig() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearProfilerConfig();
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public String getPackage() {
                return ((CalculatorGraphConfig) this.instance).getPackage();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public ByteString getPackageBytes() {
                return ((CalculatorGraphConfig) this.instance).getPackageBytes();
            }

            public Builder setPackage(String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setPackage(value);
                return this;
            }

            public Builder clearPackage() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearPackage();
                return this;
            }

            public Builder setPackageBytes(ByteString value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setPackageBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public String getType() {
                return ((CalculatorGraphConfig) this.instance).getType();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public ByteString getTypeBytes() {
                return ((CalculatorGraphConfig) this.instance).getTypeBytes();
            }

            public Builder setType(String value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setType(value);
                return this;
            }

            public Builder clearType() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearType();
                return this;
            }

            public Builder setTypeBytes(ByteString value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setTypeBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public boolean hasOptions() {
                return ((CalculatorGraphConfig) this.instance).hasOptions();
            }

            @Override // com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfigOrBuilder
            public MediaPipeOptionsProto.MediaPipeOptions getOptions() {
                return ((CalculatorGraphConfig) this.instance).getOptions();
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setOptions(value);
                return this;
            }

            public Builder setOptions(MediaPipeOptionsProto.MediaPipeOptions.Builder builderForValue) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).setOptions((MediaPipeOptionsProto.MediaPipeOptions) builderForValue.build());
                return this;
            }

            public Builder mergeOptions(MediaPipeOptionsProto.MediaPipeOptions value) {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).mergeOptions(value);
                return this;
            }

            public Builder clearOptions() {
                copyOnWrite();
                ((CalculatorGraphConfig) this.instance).clearOptions();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new CalculatorGraphConfig();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"node_", Node.class, "packetFactory_", PacketFactoryOptionsProto.PacketFactoryConfig.class, "packetGenerator_", PacketGeneratorOptionsProto.PacketGeneratorConfig.class, "numThreads_", "statusHandler_", StatusHandlerProto.StatusHandlerConfig.class, "inputStream_", "maxQueueSize_", "inputStreamHandler_", "outputStreamHandler_", "executor_", ExecutorConfig.class, "outputStream_", "inputSidePacket_", "outputSidePacket_", "profilerConfig_", "package_", "type_", "reportDeadlock_", "options_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "��\u0012����\u0001ϩ\u0012��\t\b\u0001Л\u0006Л\u0007Л\b\u0004\tЛ\nȚ\u000b\u0004\fЉ\rЉ\u000eЛ\u000fȚ\u0010Ț\u0011Ț\u0012\t\u0013Ȉ\u0014Ȉ\u0015\u0007ϩЉ", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<CalculatorGraphConfig> parser = PARSER;
                    if (parser == null) {
                        synchronized (CalculatorGraphConfig.class) {
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
            CalculatorGraphConfig defaultInstance = new CalculatorGraphConfig();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(CalculatorGraphConfig.class, defaultInstance);
        }

        public static CalculatorGraphConfig getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<CalculatorGraphConfig> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}