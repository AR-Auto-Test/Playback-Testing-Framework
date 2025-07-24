package com.google.mediapipe.tracking;

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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto.class */
public final class ModelMatrixProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedModelMatrixProtoListOrBuilder.class */
    public interface TimedModelMatrixProtoListOrBuilder extends MessageLiteOrBuilder {
        List<TimedModelMatrixProto> getModelMatrixList();

        TimedModelMatrixProto getModelMatrix(int index);

        int getModelMatrixCount();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedModelMatrixProtoOrBuilder.class */
    public interface TimedModelMatrixProtoOrBuilder extends MessageLiteOrBuilder {
        List<Float> getMatrixEntriesList();

        int getMatrixEntriesCount();

        float getMatrixEntries(int index);

        boolean hasTimeMsec();

        long getTimeMsec();

        boolean hasId();

        int getId();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedVectorProtoListOrBuilder.class */
    public interface TimedVectorProtoListOrBuilder extends MessageLiteOrBuilder {
        List<TimedVectorProto> getVectorListList();

        TimedVectorProto getVectorList(int index);

        int getVectorListCount();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedVectorProtoOrBuilder.class */
    public interface TimedVectorProtoOrBuilder extends MessageLiteOrBuilder {
        List<Float> getVectorEntriesList();

        int getVectorEntriesCount();

        float getVectorEntries(int index);

        boolean hasTimeMsec();

        long getTimeMsec();

        boolean hasId();

        int getId();
    }

    private ModelMatrixProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedModelMatrixProto.class */
    public static final class TimedModelMatrixProto extends GeneratedMessageLite<TimedModelMatrixProto, Builder> implements TimedModelMatrixProtoOrBuilder {
        private int bitField0_;
        public static final int MATRIX_ENTRIES_FIELD_NUMBER = 1;
        public static final int TIME_MSEC_FIELD_NUMBER = 2;
        private long timeMsec_;
        public static final int ID_FIELD_NUMBER = 3;
        private static final TimedModelMatrixProto DEFAULT_INSTANCE;
        private static volatile Parser<TimedModelMatrixProto> PARSER;
        private int matrixEntriesMemoizedSerializedSize = -1;
        private Internal.FloatList matrixEntries_ = emptyFloatList();
        private int id_ = -1;

        private TimedModelMatrixProto() {
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
        public List<Float> getMatrixEntriesList() {
            return this.matrixEntries_;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
        public int getMatrixEntriesCount() {
            return this.matrixEntries_.size();
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
        public float getMatrixEntries(int index) {
            return this.matrixEntries_.getFloat(index);
        }

        private void ensureMatrixEntriesIsMutable() {
            if (!this.matrixEntries_.isModifiable()) {
                this.matrixEntries_ = GeneratedMessageLite.mutableCopy(this.matrixEntries_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMatrixEntries(int index, float value) {
            ensureMatrixEntriesIsMutable();
            this.matrixEntries_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addMatrixEntries(float value) {
            ensureMatrixEntriesIsMutable();
            this.matrixEntries_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllMatrixEntries(Iterable<? extends Float> values) {
            ensureMatrixEntriesIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.matrixEntries_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMatrixEntries() {
            this.matrixEntries_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
        public boolean hasTimeMsec() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
        public long getTimeMsec() {
            return this.timeMsec_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTimeMsec(long value) {
            this.bitField0_ |= 1;
            this.timeMsec_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTimeMsec() {
            this.bitField0_ &= -2;
            this.timeMsec_ = 0L;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
        public boolean hasId() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
        public int getId() {
            return this.id_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setId(int value) {
            this.bitField0_ |= 2;
            this.id_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearId() {
            this.bitField0_ &= -3;
            this.id_ = -1;
        }

        public static TimedModelMatrixProto parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedModelMatrixProto parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedModelMatrixProto parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedModelMatrixProto parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedModelMatrixProto parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedModelMatrixProto parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedModelMatrixProto parseFrom(InputStream input) throws IOException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedModelMatrixProto parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedModelMatrixProto parseDelimitedFrom(InputStream input) throws IOException {
            return (TimedModelMatrixProto) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedModelMatrixProto parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedModelMatrixProto) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedModelMatrixProto parseFrom(CodedInputStream input) throws IOException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedModelMatrixProto parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedModelMatrixProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(TimedModelMatrixProto prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedModelMatrixProto$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<TimedModelMatrixProto, Builder> implements TimedModelMatrixProtoOrBuilder {
            private Builder() {
                super(TimedModelMatrixProto.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
            public List<Float> getMatrixEntriesList() {
                return Collections.unmodifiableList(((TimedModelMatrixProto) this.instance).getMatrixEntriesList());
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
            public int getMatrixEntriesCount() {
                return ((TimedModelMatrixProto) this.instance).getMatrixEntriesCount();
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
            public float getMatrixEntries(int index) {
                return ((TimedModelMatrixProto) this.instance).getMatrixEntries(index);
            }

            public Builder setMatrixEntries(int index, float value) {
                copyOnWrite();
                ((TimedModelMatrixProto) this.instance).setMatrixEntries(index, value);
                return this;
            }

            public Builder addMatrixEntries(float value) {
                copyOnWrite();
                ((TimedModelMatrixProto) this.instance).addMatrixEntries(value);
                return this;
            }

            public Builder addAllMatrixEntries(Iterable<? extends Float> values) {
                copyOnWrite();
                ((TimedModelMatrixProto) this.instance).addAllMatrixEntries(values);
                return this;
            }

            public Builder clearMatrixEntries() {
                copyOnWrite();
                ((TimedModelMatrixProto) this.instance).clearMatrixEntries();
                return this;
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
            public boolean hasTimeMsec() {
                return ((TimedModelMatrixProto) this.instance).hasTimeMsec();
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
            public long getTimeMsec() {
                return ((TimedModelMatrixProto) this.instance).getTimeMsec();
            }

            public Builder setTimeMsec(long value) {
                copyOnWrite();
                ((TimedModelMatrixProto) this.instance).setTimeMsec(value);
                return this;
            }

            public Builder clearTimeMsec() {
                copyOnWrite();
                ((TimedModelMatrixProto) this.instance).clearTimeMsec();
                return this;
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
            public boolean hasId() {
                return ((TimedModelMatrixProto) this.instance).hasId();
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoOrBuilder
            public int getId() {
                return ((TimedModelMatrixProto) this.instance).getId();
            }

            public Builder setId(int value) {
                copyOnWrite();
                ((TimedModelMatrixProto) this.instance).setId(value);
                return this;
            }

            public Builder clearId() {
                copyOnWrite();
                ((TimedModelMatrixProto) this.instance).clearId();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new TimedModelMatrixProto();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "matrixEntries_", "timeMsec_", "id_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0003��\u0001\u0001\u0003\u0003��\u0001��\u0001$\u0002\u0002��\u0003\u0004\u0001", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<TimedModelMatrixProto> parser = PARSER;
                    if (parser == null) {
                        synchronized (TimedModelMatrixProto.class) {
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
            TimedModelMatrixProto defaultInstance = new TimedModelMatrixProto();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(TimedModelMatrixProto.class, defaultInstance);
        }

        public static TimedModelMatrixProto getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<TimedModelMatrixProto> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedModelMatrixProtoList.class */
    public static final class TimedModelMatrixProtoList extends GeneratedMessageLite<TimedModelMatrixProtoList, Builder> implements TimedModelMatrixProtoListOrBuilder {
        public static final int MODEL_MATRIX_FIELD_NUMBER = 1;
        private Internal.ProtobufList<TimedModelMatrixProto> modelMatrix_ = emptyProtobufList();
        private static final TimedModelMatrixProtoList DEFAULT_INSTANCE;
        private static volatile Parser<TimedModelMatrixProtoList> PARSER;

        private TimedModelMatrixProtoList() {
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoListOrBuilder
        public List<TimedModelMatrixProto> getModelMatrixList() {
            return this.modelMatrix_;
        }

        public List<? extends TimedModelMatrixProtoOrBuilder> getModelMatrixOrBuilderList() {
            return this.modelMatrix_;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoListOrBuilder
        public int getModelMatrixCount() {
            return this.modelMatrix_.size();
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoListOrBuilder
        public TimedModelMatrixProto getModelMatrix(int index) {
            return this.modelMatrix_.get(index);
        }

        public TimedModelMatrixProtoOrBuilder getModelMatrixOrBuilder(int index) {
            return this.modelMatrix_.get(index);
        }

        private void ensureModelMatrixIsMutable() {
            if (!this.modelMatrix_.isModifiable()) {
                this.modelMatrix_ = GeneratedMessageLite.mutableCopy(this.modelMatrix_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setModelMatrix(int index, TimedModelMatrixProto value) {
            value.getClass();
            ensureModelMatrixIsMutable();
            this.modelMatrix_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addModelMatrix(TimedModelMatrixProto value) {
            value.getClass();
            ensureModelMatrixIsMutable();
            this.modelMatrix_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addModelMatrix(int index, TimedModelMatrixProto value) {
            value.getClass();
            ensureModelMatrixIsMutable();
            this.modelMatrix_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllModelMatrix(Iterable<? extends TimedModelMatrixProto> values) {
            ensureModelMatrixIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.modelMatrix_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearModelMatrix() {
            this.modelMatrix_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeModelMatrix(int index) {
            ensureModelMatrixIsMutable();
            this.modelMatrix_.remove(index);
        }

        public static TimedModelMatrixProtoList parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedModelMatrixProtoList parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedModelMatrixProtoList parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedModelMatrixProtoList parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedModelMatrixProtoList parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedModelMatrixProtoList parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedModelMatrixProtoList parseFrom(InputStream input) throws IOException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedModelMatrixProtoList parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedModelMatrixProtoList parseDelimitedFrom(InputStream input) throws IOException {
            return (TimedModelMatrixProtoList) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedModelMatrixProtoList parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedModelMatrixProtoList) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedModelMatrixProtoList parseFrom(CodedInputStream input) throws IOException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedModelMatrixProtoList parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedModelMatrixProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(TimedModelMatrixProtoList prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedModelMatrixProtoList$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<TimedModelMatrixProtoList, Builder> implements TimedModelMatrixProtoListOrBuilder {
            private Builder() {
                super(TimedModelMatrixProtoList.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoListOrBuilder
            public List<TimedModelMatrixProto> getModelMatrixList() {
                return Collections.unmodifiableList(((TimedModelMatrixProtoList) this.instance).getModelMatrixList());
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoListOrBuilder
            public int getModelMatrixCount() {
                return ((TimedModelMatrixProtoList) this.instance).getModelMatrixCount();
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedModelMatrixProtoListOrBuilder
            public TimedModelMatrixProto getModelMatrix(int index) {
                return ((TimedModelMatrixProtoList) this.instance).getModelMatrix(index);
            }

            public Builder setModelMatrix(int index, TimedModelMatrixProto value) {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).setModelMatrix(index, value);
                return this;
            }

            public Builder setModelMatrix(int index, TimedModelMatrixProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).setModelMatrix(index, builderForValue.build());
                return this;
            }

            public Builder addModelMatrix(TimedModelMatrixProto value) {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).addModelMatrix(value);
                return this;
            }

            public Builder addModelMatrix(int index, TimedModelMatrixProto value) {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).addModelMatrix(index, value);
                return this;
            }

            public Builder addModelMatrix(TimedModelMatrixProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).addModelMatrix(builderForValue.build());
                return this;
            }

            public Builder addModelMatrix(int index, TimedModelMatrixProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).addModelMatrix(index, builderForValue.build());
                return this;
            }

            public Builder addAllModelMatrix(Iterable<? extends TimedModelMatrixProto> values) {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).addAllModelMatrix(values);
                return this;
            }

            public Builder clearModelMatrix() {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).clearModelMatrix();
                return this;
            }

            public Builder removeModelMatrix(int index) {
                copyOnWrite();
                ((TimedModelMatrixProtoList) this.instance).removeModelMatrix(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new TimedModelMatrixProtoList();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"modelMatrix_", TimedModelMatrixProto.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<TimedModelMatrixProtoList> parser = PARSER;
                    if (parser == null) {
                        synchronized (TimedModelMatrixProtoList.class) {
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
            TimedModelMatrixProtoList defaultInstance = new TimedModelMatrixProtoList();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(TimedModelMatrixProtoList.class, defaultInstance);
        }

        public static TimedModelMatrixProtoList getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<TimedModelMatrixProtoList> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedVectorProto.class */
    public static final class TimedVectorProto extends GeneratedMessageLite<TimedVectorProto, Builder> implements TimedVectorProtoOrBuilder {
        private int bitField0_;
        public static final int VECTOR_ENTRIES_FIELD_NUMBER = 1;
        public static final int TIME_MSEC_FIELD_NUMBER = 2;
        private long timeMsec_;
        public static final int ID_FIELD_NUMBER = 3;
        private static final TimedVectorProto DEFAULT_INSTANCE;
        private static volatile Parser<TimedVectorProto> PARSER;
        private int vectorEntriesMemoizedSerializedSize = -1;
        private Internal.FloatList vectorEntries_ = emptyFloatList();
        private int id_ = -1;

        private TimedVectorProto() {
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
        public List<Float> getVectorEntriesList() {
            return this.vectorEntries_;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
        public int getVectorEntriesCount() {
            return this.vectorEntries_.size();
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
        public float getVectorEntries(int index) {
            return this.vectorEntries_.getFloat(index);
        }

        private void ensureVectorEntriesIsMutable() {
            if (!this.vectorEntries_.isModifiable()) {
                this.vectorEntries_ = GeneratedMessageLite.mutableCopy(this.vectorEntries_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setVectorEntries(int index, float value) {
            ensureVectorEntriesIsMutable();
            this.vectorEntries_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addVectorEntries(float value) {
            ensureVectorEntriesIsMutable();
            this.vectorEntries_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllVectorEntries(Iterable<? extends Float> values) {
            ensureVectorEntriesIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.vectorEntries_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearVectorEntries() {
            this.vectorEntries_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
        public boolean hasTimeMsec() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
        public long getTimeMsec() {
            return this.timeMsec_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTimeMsec(long value) {
            this.bitField0_ |= 1;
            this.timeMsec_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTimeMsec() {
            this.bitField0_ &= -2;
            this.timeMsec_ = 0L;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
        public boolean hasId() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
        public int getId() {
            return this.id_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setId(int value) {
            this.bitField0_ |= 2;
            this.id_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearId() {
            this.bitField0_ &= -3;
            this.id_ = -1;
        }

        public static TimedVectorProto parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedVectorProto parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedVectorProto parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedVectorProto parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedVectorProto parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedVectorProto parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedVectorProto parseFrom(InputStream input) throws IOException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedVectorProto parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedVectorProto parseDelimitedFrom(InputStream input) throws IOException {
            return (TimedVectorProto) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedVectorProto parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedVectorProto) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedVectorProto parseFrom(CodedInputStream input) throws IOException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedVectorProto parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedVectorProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(TimedVectorProto prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedVectorProto$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<TimedVectorProto, Builder> implements TimedVectorProtoOrBuilder {
            private Builder() {
                super(TimedVectorProto.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
            public List<Float> getVectorEntriesList() {
                return Collections.unmodifiableList(((TimedVectorProto) this.instance).getVectorEntriesList());
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
            public int getVectorEntriesCount() {
                return ((TimedVectorProto) this.instance).getVectorEntriesCount();
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
            public float getVectorEntries(int index) {
                return ((TimedVectorProto) this.instance).getVectorEntries(index);
            }

            public Builder setVectorEntries(int index, float value) {
                copyOnWrite();
                ((TimedVectorProto) this.instance).setVectorEntries(index, value);
                return this;
            }

            public Builder addVectorEntries(float value) {
                copyOnWrite();
                ((TimedVectorProto) this.instance).addVectorEntries(value);
                return this;
            }

            public Builder addAllVectorEntries(Iterable<? extends Float> values) {
                copyOnWrite();
                ((TimedVectorProto) this.instance).addAllVectorEntries(values);
                return this;
            }

            public Builder clearVectorEntries() {
                copyOnWrite();
                ((TimedVectorProto) this.instance).clearVectorEntries();
                return this;
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
            public boolean hasTimeMsec() {
                return ((TimedVectorProto) this.instance).hasTimeMsec();
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
            public long getTimeMsec() {
                return ((TimedVectorProto) this.instance).getTimeMsec();
            }

            public Builder setTimeMsec(long value) {
                copyOnWrite();
                ((TimedVectorProto) this.instance).setTimeMsec(value);
                return this;
            }

            public Builder clearTimeMsec() {
                copyOnWrite();
                ((TimedVectorProto) this.instance).clearTimeMsec();
                return this;
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
            public boolean hasId() {
                return ((TimedVectorProto) this.instance).hasId();
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoOrBuilder
            public int getId() {
                return ((TimedVectorProto) this.instance).getId();
            }

            public Builder setId(int value) {
                copyOnWrite();
                ((TimedVectorProto) this.instance).setId(value);
                return this;
            }

            public Builder clearId() {
                copyOnWrite();
                ((TimedVectorProto) this.instance).clearId();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new TimedVectorProto();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "vectorEntries_", "timeMsec_", "id_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0003��\u0001\u0001\u0003\u0003��\u0001��\u0001$\u0002\u0002��\u0003\u0004\u0001", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<TimedVectorProto> parser = PARSER;
                    if (parser == null) {
                        synchronized (TimedVectorProto.class) {
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
            TimedVectorProto defaultInstance = new TimedVectorProto();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(TimedVectorProto.class, defaultInstance);
        }

        public static TimedVectorProto getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<TimedVectorProto> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedVectorProtoList.class */
    public static final class TimedVectorProtoList extends GeneratedMessageLite<TimedVectorProtoList, Builder> implements TimedVectorProtoListOrBuilder {
        public static final int VECTOR_LIST_FIELD_NUMBER = 1;
        private Internal.ProtobufList<TimedVectorProto> vectorList_ = emptyProtobufList();
        private static final TimedVectorProtoList DEFAULT_INSTANCE;
        private static volatile Parser<TimedVectorProtoList> PARSER;

        private TimedVectorProtoList() {
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoListOrBuilder
        public List<TimedVectorProto> getVectorListList() {
            return this.vectorList_;
        }

        public List<? extends TimedVectorProtoOrBuilder> getVectorListOrBuilderList() {
            return this.vectorList_;
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoListOrBuilder
        public int getVectorListCount() {
            return this.vectorList_.size();
        }

        @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoListOrBuilder
        public TimedVectorProto getVectorList(int index) {
            return this.vectorList_.get(index);
        }

        public TimedVectorProtoOrBuilder getVectorListOrBuilder(int index) {
            return this.vectorList_.get(index);
        }

        private void ensureVectorListIsMutable() {
            if (!this.vectorList_.isModifiable()) {
                this.vectorList_ = GeneratedMessageLite.mutableCopy(this.vectorList_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setVectorList(int index, TimedVectorProto value) {
            value.getClass();
            ensureVectorListIsMutable();
            this.vectorList_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addVectorList(TimedVectorProto value) {
            value.getClass();
            ensureVectorListIsMutable();
            this.vectorList_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addVectorList(int index, TimedVectorProto value) {
            value.getClass();
            ensureVectorListIsMutable();
            this.vectorList_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllVectorList(Iterable<? extends TimedVectorProto> values) {
            ensureVectorListIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.vectorList_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearVectorList() {
            this.vectorList_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeVectorList(int index) {
            ensureVectorListIsMutable();
            this.vectorList_.remove(index);
        }

        public static TimedVectorProtoList parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedVectorProtoList parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedVectorProtoList parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedVectorProtoList parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedVectorProtoList parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedVectorProtoList parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedVectorProtoList parseFrom(InputStream input) throws IOException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedVectorProtoList parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedVectorProtoList parseDelimitedFrom(InputStream input) throws IOException {
            return (TimedVectorProtoList) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedVectorProtoList parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedVectorProtoList) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedVectorProtoList parseFrom(CodedInputStream input) throws IOException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedVectorProtoList parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedVectorProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(TimedVectorProtoList prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/ModelMatrixProto$TimedVectorProtoList$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<TimedVectorProtoList, Builder> implements TimedVectorProtoListOrBuilder {
            private Builder() {
                super(TimedVectorProtoList.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoListOrBuilder
            public List<TimedVectorProto> getVectorListList() {
                return Collections.unmodifiableList(((TimedVectorProtoList) this.instance).getVectorListList());
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoListOrBuilder
            public int getVectorListCount() {
                return ((TimedVectorProtoList) this.instance).getVectorListCount();
            }

            @Override // com.google.mediapipe.tracking.ModelMatrixProto.TimedVectorProtoListOrBuilder
            public TimedVectorProto getVectorList(int index) {
                return ((TimedVectorProtoList) this.instance).getVectorList(index);
            }

            public Builder setVectorList(int index, TimedVectorProto value) {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).setVectorList(index, value);
                return this;
            }

            public Builder setVectorList(int index, TimedVectorProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).setVectorList(index, builderForValue.build());
                return this;
            }

            public Builder addVectorList(TimedVectorProto value) {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).addVectorList(value);
                return this;
            }

            public Builder addVectorList(int index, TimedVectorProto value) {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).addVectorList(index, value);
                return this;
            }

            public Builder addVectorList(TimedVectorProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).addVectorList(builderForValue.build());
                return this;
            }

            public Builder addVectorList(int index, TimedVectorProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).addVectorList(index, builderForValue.build());
                return this;
            }

            public Builder addAllVectorList(Iterable<? extends TimedVectorProto> values) {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).addAllVectorList(values);
                return this;
            }

            public Builder clearVectorList() {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).clearVectorList();
                return this;
            }

            public Builder removeVectorList(int index) {
                copyOnWrite();
                ((TimedVectorProtoList) this.instance).removeVectorList(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new TimedVectorProtoList();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"vectorList_", TimedVectorProto.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<TimedVectorProtoList> parser = PARSER;
                    if (parser == null) {
                        synchronized (TimedVectorProtoList.class) {
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
            TimedVectorProtoList defaultInstance = new TimedVectorProtoList();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(TimedVectorProtoList.class, defaultInstance);
        }

        public static TimedVectorProtoList getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<TimedVectorProtoList> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}