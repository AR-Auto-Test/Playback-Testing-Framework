package com.google.mediapipe.formats.annotation.proto;

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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/annotation/proto/RasterizationProto.class */
public final class RasterizationProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/annotation/proto/RasterizationProto$RasterizationOrBuilder.class */
    public interface RasterizationOrBuilder extends MessageLiteOrBuilder {
        List<Rasterization.Interval> getIntervalList();

        Rasterization.Interval getInterval(int index);

        int getIntervalCount();
    }

    private RasterizationProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/annotation/proto/RasterizationProto$Rasterization.class */
    public static final class Rasterization extends GeneratedMessageLite<Rasterization, Builder> implements RasterizationOrBuilder {
        public static final int INTERVAL_FIELD_NUMBER = 1;
        private static final Rasterization DEFAULT_INSTANCE;
        private static volatile Parser<Rasterization> PARSER;
        private byte memoizedIsInitialized = 2;
        private Internal.ProtobufList<Interval> interval_ = emptyProtobufList();

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/annotation/proto/RasterizationProto$Rasterization$IntervalOrBuilder.class */
        public interface IntervalOrBuilder extends MessageLiteOrBuilder {
            boolean hasY();

            int getY();

            boolean hasLeftX();

            int getLeftX();

            boolean hasRightX();

            int getRightX();
        }

        private Rasterization() {
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/annotation/proto/RasterizationProto$Rasterization$Interval.class */
        public static final class Interval extends GeneratedMessageLite<Interval, Builder> implements IntervalOrBuilder {
            private int bitField0_;
            public static final int Y_FIELD_NUMBER = 1;
            private int y_;
            public static final int LEFT_X_FIELD_NUMBER = 2;
            private int leftX_;
            public static final int RIGHT_X_FIELD_NUMBER = 3;
            private int rightX_;
            private byte memoizedIsInitialized = 2;
            private static final Interval DEFAULT_INSTANCE;
            private static volatile Parser<Interval> PARSER;

            private Interval() {
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
            public boolean hasY() {
                return (this.bitField0_ & 1) != 0;
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
            public int getY() {
                return this.y_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setY(int value) {
                this.bitField0_ |= 1;
                this.y_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearY() {
                this.bitField0_ &= -2;
                this.y_ = 0;
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
            public boolean hasLeftX() {
                return (this.bitField0_ & 2) != 0;
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
            public int getLeftX() {
                return this.leftX_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setLeftX(int value) {
                this.bitField0_ |= 2;
                this.leftX_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearLeftX() {
                this.bitField0_ &= -3;
                this.leftX_ = 0;
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
            public boolean hasRightX() {
                return (this.bitField0_ & 4) != 0;
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
            public int getRightX() {
                return this.rightX_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setRightX(int value) {
                this.bitField0_ |= 4;
                this.rightX_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearRightX() {
                this.bitField0_ &= -5;
                this.rightX_ = 0;
            }

            public static Interval parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Interval parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Interval parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Interval parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Interval parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Interval parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Interval parseFrom(InputStream input) throws IOException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static Interval parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Interval parseDelimitedFrom(InputStream input) throws IOException {
                return (Interval) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static Interval parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Interval) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Interval parseFrom(CodedInputStream input) throws IOException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static Interval parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Interval) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(Interval prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/annotation/proto/RasterizationProto$Rasterization$Interval$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<Interval, Builder> implements IntervalOrBuilder {
                private Builder() {
                    super(Interval.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
                public boolean hasY() {
                    return ((Interval) this.instance).hasY();
                }

                @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
                public int getY() {
                    return ((Interval) this.instance).getY();
                }

                public Builder setY(int value) {
                    copyOnWrite();
                    ((Interval) this.instance).setY(value);
                    return this;
                }

                public Builder clearY() {
                    copyOnWrite();
                    ((Interval) this.instance).clearY();
                    return this;
                }

                @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
                public boolean hasLeftX() {
                    return ((Interval) this.instance).hasLeftX();
                }

                @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
                public int getLeftX() {
                    return ((Interval) this.instance).getLeftX();
                }

                public Builder setLeftX(int value) {
                    copyOnWrite();
                    ((Interval) this.instance).setLeftX(value);
                    return this;
                }

                public Builder clearLeftX() {
                    copyOnWrite();
                    ((Interval) this.instance).clearLeftX();
                    return this;
                }

                @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
                public boolean hasRightX() {
                    return ((Interval) this.instance).hasRightX();
                }

                @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.Rasterization.IntervalOrBuilder
                public int getRightX() {
                    return ((Interval) this.instance).getRightX();
                }

                public Builder setRightX(int value) {
                    copyOnWrite();
                    ((Interval) this.instance).setRightX(value);
                    return this;
                }

                public Builder clearRightX() {
                    copyOnWrite();
                    ((Interval) this.instance).clearRightX();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new Interval();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"bitField0_", "y_", "leftX_", "rightX_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0003��\u0001\u0001\u0003\u0003����\u0003\u0001Ԅ��\u0002Ԅ\u0001\u0003Ԅ\u0002", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<Interval> parser = PARSER;
                        if (parser == null) {
                            synchronized (Interval.class) {
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
                Interval defaultInstance = new Interval();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(Interval.class, defaultInstance);
            }

            public static Interval getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<Interval> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.RasterizationOrBuilder
        public List<Interval> getIntervalList() {
            return this.interval_;
        }

        public List<? extends IntervalOrBuilder> getIntervalOrBuilderList() {
            return this.interval_;
        }

        @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.RasterizationOrBuilder
        public int getIntervalCount() {
            return this.interval_.size();
        }

        @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.RasterizationOrBuilder
        public Interval getInterval(int index) {
            return this.interval_.get(index);
        }

        public IntervalOrBuilder getIntervalOrBuilder(int index) {
            return this.interval_.get(index);
        }

        private void ensureIntervalIsMutable() {
            if (!this.interval_.isModifiable()) {
                this.interval_ = GeneratedMessageLite.mutableCopy(this.interval_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInterval(int index, Interval value) {
            value.getClass();
            ensureIntervalIsMutable();
            this.interval_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInterval(Interval value) {
            value.getClass();
            ensureIntervalIsMutable();
            this.interval_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInterval(int index, Interval value) {
            value.getClass();
            ensureIntervalIsMutable();
            this.interval_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllInterval(Iterable<? extends Interval> values) {
            ensureIntervalIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.interval_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInterval() {
            this.interval_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeInterval(int index) {
            ensureIntervalIsMutable();
            this.interval_.remove(index);
        }

        public static Rasterization parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Rasterization parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Rasterization parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Rasterization parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Rasterization parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Rasterization parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Rasterization parseFrom(InputStream input) throws IOException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static Rasterization parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Rasterization parseDelimitedFrom(InputStream input) throws IOException {
            return (Rasterization) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static Rasterization parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Rasterization) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Rasterization parseFrom(CodedInputStream input) throws IOException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static Rasterization parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Rasterization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(Rasterization prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/annotation/proto/RasterizationProto$Rasterization$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<Rasterization, Builder> implements RasterizationOrBuilder {
            private Builder() {
                super(Rasterization.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.RasterizationOrBuilder
            public List<Interval> getIntervalList() {
                return Collections.unmodifiableList(((Rasterization) this.instance).getIntervalList());
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.RasterizationOrBuilder
            public int getIntervalCount() {
                return ((Rasterization) this.instance).getIntervalCount();
            }

            @Override // com.google.mediapipe.formats.annotation.proto.RasterizationProto.RasterizationOrBuilder
            public Interval getInterval(int index) {
                return ((Rasterization) this.instance).getInterval(index);
            }

            public Builder setInterval(int index, Interval value) {
                copyOnWrite();
                ((Rasterization) this.instance).setInterval(index, value);
                return this;
            }

            public Builder setInterval(int index, Interval.Builder builderForValue) {
                copyOnWrite();
                ((Rasterization) this.instance).setInterval(index, builderForValue.build());
                return this;
            }

            public Builder addInterval(Interval value) {
                copyOnWrite();
                ((Rasterization) this.instance).addInterval(value);
                return this;
            }

            public Builder addInterval(int index, Interval value) {
                copyOnWrite();
                ((Rasterization) this.instance).addInterval(index, value);
                return this;
            }

            public Builder addInterval(Interval.Builder builderForValue) {
                copyOnWrite();
                ((Rasterization) this.instance).addInterval(builderForValue.build());
                return this;
            }

            public Builder addInterval(int index, Interval.Builder builderForValue) {
                copyOnWrite();
                ((Rasterization) this.instance).addInterval(index, builderForValue.build());
                return this;
            }

            public Builder addAllInterval(Iterable<? extends Interval> values) {
                copyOnWrite();
                ((Rasterization) this.instance).addAllInterval(values);
                return this;
            }

            public Builder clearInterval() {
                copyOnWrite();
                ((Rasterization) this.instance).clearInterval();
                return this;
            }

            public Builder removeInterval(int index) {
                copyOnWrite();
                ((Rasterization) this.instance).removeInterval(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new Rasterization();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"interval_", Interval.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001\u0001\u0001Л", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<Rasterization> parser = PARSER;
                    if (parser == null) {
                        synchronized (Rasterization.class) {
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
            Rasterization defaultInstance = new Rasterization();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(Rasterization.class, defaultInstance);
        }

        public static Rasterization getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<Rasterization> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}