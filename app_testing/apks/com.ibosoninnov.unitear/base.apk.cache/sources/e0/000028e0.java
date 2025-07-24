package com.google.mediapipe.formats.proto;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto.class */
public final class LandmarkProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$LandmarkListOrBuilder.class */
    public interface LandmarkListOrBuilder extends MessageLiteOrBuilder {
        List<Landmark> getLandmarkList();

        Landmark getLandmark(int index);

        int getLandmarkCount();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$LandmarkOrBuilder.class */
    public interface LandmarkOrBuilder extends MessageLiteOrBuilder {
        boolean hasX();

        float getX();

        boolean hasY();

        float getY();

        boolean hasZ();

        float getZ();

        boolean hasVisibility();

        float getVisibility();

        boolean hasPresence();

        float getPresence();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$NormalizedLandmarkListOrBuilder.class */
    public interface NormalizedLandmarkListOrBuilder extends MessageLiteOrBuilder {
        List<NormalizedLandmark> getLandmarkList();

        NormalizedLandmark getLandmark(int index);

        int getLandmarkCount();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$NormalizedLandmarkOrBuilder.class */
    public interface NormalizedLandmarkOrBuilder extends MessageLiteOrBuilder {
        boolean hasX();

        float getX();

        boolean hasY();

        float getY();

        boolean hasZ();

        float getZ();

        boolean hasVisibility();

        float getVisibility();

        boolean hasPresence();

        float getPresence();
    }

    private LandmarkProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$Landmark.class */
    public static final class Landmark extends GeneratedMessageLite<Landmark, Builder> implements LandmarkOrBuilder {
        private int bitField0_;
        public static final int X_FIELD_NUMBER = 1;
        private float x_;
        public static final int Y_FIELD_NUMBER = 2;
        private float y_;
        public static final int Z_FIELD_NUMBER = 3;
        private float z_;
        public static final int VISIBILITY_FIELD_NUMBER = 4;
        private float visibility_;
        public static final int PRESENCE_FIELD_NUMBER = 5;
        private float presence_;
        private static final Landmark DEFAULT_INSTANCE;
        private static volatile Parser<Landmark> PARSER;

        private Landmark() {
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public boolean hasX() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public float getX() {
            return this.x_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setX(float value) {
            this.bitField0_ |= 1;
            this.x_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearX() {
            this.bitField0_ &= -2;
            this.x_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public boolean hasY() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public float getY() {
            return this.y_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setY(float value) {
            this.bitField0_ |= 2;
            this.y_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearY() {
            this.bitField0_ &= -3;
            this.y_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public boolean hasZ() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public float getZ() {
            return this.z_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setZ(float value) {
            this.bitField0_ |= 4;
            this.z_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearZ() {
            this.bitField0_ &= -5;
            this.z_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public boolean hasVisibility() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public float getVisibility() {
            return this.visibility_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setVisibility(float value) {
            this.bitField0_ |= 8;
            this.visibility_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearVisibility() {
            this.bitField0_ &= -9;
            this.visibility_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public boolean hasPresence() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
        public float getPresence() {
            return this.presence_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPresence(float value) {
            this.bitField0_ |= 16;
            this.presence_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPresence() {
            this.bitField0_ &= -17;
            this.presence_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        public static Landmark parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Landmark parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Landmark parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Landmark parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Landmark parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Landmark parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Landmark parseFrom(InputStream input) throws IOException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static Landmark parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Landmark parseDelimitedFrom(InputStream input) throws IOException {
            return (Landmark) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static Landmark parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Landmark) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Landmark parseFrom(CodedInputStream input) throws IOException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static Landmark parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Landmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(Landmark prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$Landmark$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<Landmark, Builder> implements LandmarkOrBuilder {
            private Builder() {
                super(Landmark.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public boolean hasX() {
                return ((Landmark) this.instance).hasX();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public float getX() {
                return ((Landmark) this.instance).getX();
            }

            public Builder setX(float value) {
                copyOnWrite();
                ((Landmark) this.instance).setX(value);
                return this;
            }

            public Builder clearX() {
                copyOnWrite();
                ((Landmark) this.instance).clearX();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public boolean hasY() {
                return ((Landmark) this.instance).hasY();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public float getY() {
                return ((Landmark) this.instance).getY();
            }

            public Builder setY(float value) {
                copyOnWrite();
                ((Landmark) this.instance).setY(value);
                return this;
            }

            public Builder clearY() {
                copyOnWrite();
                ((Landmark) this.instance).clearY();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public boolean hasZ() {
                return ((Landmark) this.instance).hasZ();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public float getZ() {
                return ((Landmark) this.instance).getZ();
            }

            public Builder setZ(float value) {
                copyOnWrite();
                ((Landmark) this.instance).setZ(value);
                return this;
            }

            public Builder clearZ() {
                copyOnWrite();
                ((Landmark) this.instance).clearZ();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public boolean hasVisibility() {
                return ((Landmark) this.instance).hasVisibility();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public float getVisibility() {
                return ((Landmark) this.instance).getVisibility();
            }

            public Builder setVisibility(float value) {
                copyOnWrite();
                ((Landmark) this.instance).setVisibility(value);
                return this;
            }

            public Builder clearVisibility() {
                copyOnWrite();
                ((Landmark) this.instance).clearVisibility();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public boolean hasPresence() {
                return ((Landmark) this.instance).hasPresence();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkOrBuilder
            public float getPresence() {
                return ((Landmark) this.instance).getPresence();
            }

            public Builder setPresence(float value) {
                copyOnWrite();
                ((Landmark) this.instance).setPresence(value);
                return this;
            }

            public Builder clearPresence() {
                copyOnWrite();
                ((Landmark) this.instance).clearPresence();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new Landmark();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "x_", "y_", "z_", "visibility_", "presence_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0005��\u0001\u0001\u0005\u0005������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003\u0005\u0001\u0004", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<Landmark> parser = PARSER;
                    if (parser == null) {
                        synchronized (Landmark.class) {
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
            Landmark defaultInstance = new Landmark();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(Landmark.class, defaultInstance);
        }

        public static Landmark getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<Landmark> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$LandmarkList.class */
    public static final class LandmarkList extends GeneratedMessageLite<LandmarkList, Builder> implements LandmarkListOrBuilder {
        public static final int LANDMARK_FIELD_NUMBER = 1;
        private Internal.ProtobufList<Landmark> landmark_ = emptyProtobufList();
        private static final LandmarkList DEFAULT_INSTANCE;
        private static volatile Parser<LandmarkList> PARSER;

        private LandmarkList() {
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkListOrBuilder
        public List<Landmark> getLandmarkList() {
            return this.landmark_;
        }

        public List<? extends LandmarkOrBuilder> getLandmarkOrBuilderList() {
            return this.landmark_;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkListOrBuilder
        public int getLandmarkCount() {
            return this.landmark_.size();
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkListOrBuilder
        public Landmark getLandmark(int index) {
            return this.landmark_.get(index);
        }

        public LandmarkOrBuilder getLandmarkOrBuilder(int index) {
            return this.landmark_.get(index);
        }

        private void ensureLandmarkIsMutable() {
            if (!this.landmark_.isModifiable()) {
                this.landmark_ = GeneratedMessageLite.mutableCopy(this.landmark_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setLandmark(int index, Landmark value) {
            value.getClass();
            ensureLandmarkIsMutable();
            this.landmark_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addLandmark(Landmark value) {
            value.getClass();
            ensureLandmarkIsMutable();
            this.landmark_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addLandmark(int index, Landmark value) {
            value.getClass();
            ensureLandmarkIsMutable();
            this.landmark_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllLandmark(Iterable<? extends Landmark> values) {
            ensureLandmarkIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.landmark_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearLandmark() {
            this.landmark_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeLandmark(int index) {
            ensureLandmarkIsMutable();
            this.landmark_.remove(index);
        }

        public static LandmarkList parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LandmarkList parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LandmarkList parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LandmarkList parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LandmarkList parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LandmarkList parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LandmarkList parseFrom(InputStream input) throws IOException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static LandmarkList parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static LandmarkList parseDelimitedFrom(InputStream input) throws IOException {
            return (LandmarkList) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static LandmarkList parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LandmarkList) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static LandmarkList parseFrom(CodedInputStream input) throws IOException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static LandmarkList parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(LandmarkList prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$LandmarkList$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<LandmarkList, Builder> implements LandmarkListOrBuilder {
            private Builder() {
                super(LandmarkList.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkListOrBuilder
            public List<Landmark> getLandmarkList() {
                return Collections.unmodifiableList(((LandmarkList) this.instance).getLandmarkList());
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkListOrBuilder
            public int getLandmarkCount() {
                return ((LandmarkList) this.instance).getLandmarkCount();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.LandmarkListOrBuilder
            public Landmark getLandmark(int index) {
                return ((LandmarkList) this.instance).getLandmark(index);
            }

            public Builder setLandmark(int index, Landmark value) {
                copyOnWrite();
                ((LandmarkList) this.instance).setLandmark(index, value);
                return this;
            }

            public Builder setLandmark(int index, Landmark.Builder builderForValue) {
                copyOnWrite();
                ((LandmarkList) this.instance).setLandmark(index, builderForValue.build());
                return this;
            }

            public Builder addLandmark(Landmark value) {
                copyOnWrite();
                ((LandmarkList) this.instance).addLandmark(value);
                return this;
            }

            public Builder addLandmark(int index, Landmark value) {
                copyOnWrite();
                ((LandmarkList) this.instance).addLandmark(index, value);
                return this;
            }

            public Builder addLandmark(Landmark.Builder builderForValue) {
                copyOnWrite();
                ((LandmarkList) this.instance).addLandmark(builderForValue.build());
                return this;
            }

            public Builder addLandmark(int index, Landmark.Builder builderForValue) {
                copyOnWrite();
                ((LandmarkList) this.instance).addLandmark(index, builderForValue.build());
                return this;
            }

            public Builder addAllLandmark(Iterable<? extends Landmark> values) {
                copyOnWrite();
                ((LandmarkList) this.instance).addAllLandmark(values);
                return this;
            }

            public Builder clearLandmark() {
                copyOnWrite();
                ((LandmarkList) this.instance).clearLandmark();
                return this;
            }

            public Builder removeLandmark(int index) {
                copyOnWrite();
                ((LandmarkList) this.instance).removeLandmark(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new LandmarkList();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"landmark_", Landmark.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<LandmarkList> parser = PARSER;
                    if (parser == null) {
                        synchronized (LandmarkList.class) {
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
            LandmarkList defaultInstance = new LandmarkList();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(LandmarkList.class, defaultInstance);
        }

        public static LandmarkList getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<LandmarkList> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$NormalizedLandmark.class */
    public static final class NormalizedLandmark extends GeneratedMessageLite<NormalizedLandmark, Builder> implements NormalizedLandmarkOrBuilder {
        private int bitField0_;
        public static final int X_FIELD_NUMBER = 1;
        private float x_;
        public static final int Y_FIELD_NUMBER = 2;
        private float y_;
        public static final int Z_FIELD_NUMBER = 3;
        private float z_;
        public static final int VISIBILITY_FIELD_NUMBER = 4;
        private float visibility_;
        public static final int PRESENCE_FIELD_NUMBER = 5;
        private float presence_;
        private static final NormalizedLandmark DEFAULT_INSTANCE;
        private static volatile Parser<NormalizedLandmark> PARSER;

        private NormalizedLandmark() {
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public boolean hasX() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public float getX() {
            return this.x_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setX(float value) {
            this.bitField0_ |= 1;
            this.x_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearX() {
            this.bitField0_ &= -2;
            this.x_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public boolean hasY() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public float getY() {
            return this.y_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setY(float value) {
            this.bitField0_ |= 2;
            this.y_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearY() {
            this.bitField0_ &= -3;
            this.y_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public boolean hasZ() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public float getZ() {
            return this.z_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setZ(float value) {
            this.bitField0_ |= 4;
            this.z_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearZ() {
            this.bitField0_ &= -5;
            this.z_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public boolean hasVisibility() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public float getVisibility() {
            return this.visibility_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setVisibility(float value) {
            this.bitField0_ |= 8;
            this.visibility_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearVisibility() {
            this.bitField0_ &= -9;
            this.visibility_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public boolean hasPresence() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
        public float getPresence() {
            return this.presence_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPresence(float value) {
            this.bitField0_ |= 16;
            this.presence_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPresence() {
            this.bitField0_ &= -17;
            this.presence_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        public static NormalizedLandmark parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static NormalizedLandmark parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static NormalizedLandmark parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static NormalizedLandmark parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static NormalizedLandmark parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static NormalizedLandmark parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static NormalizedLandmark parseFrom(InputStream input) throws IOException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static NormalizedLandmark parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static NormalizedLandmark parseDelimitedFrom(InputStream input) throws IOException {
            return (NormalizedLandmark) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static NormalizedLandmark parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (NormalizedLandmark) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static NormalizedLandmark parseFrom(CodedInputStream input) throws IOException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static NormalizedLandmark parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (NormalizedLandmark) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(NormalizedLandmark prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$NormalizedLandmark$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<NormalizedLandmark, Builder> implements NormalizedLandmarkOrBuilder {
            private Builder() {
                super(NormalizedLandmark.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public boolean hasX() {
                return ((NormalizedLandmark) this.instance).hasX();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public float getX() {
                return ((NormalizedLandmark) this.instance).getX();
            }

            public Builder setX(float value) {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).setX(value);
                return this;
            }

            public Builder clearX() {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).clearX();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public boolean hasY() {
                return ((NormalizedLandmark) this.instance).hasY();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public float getY() {
                return ((NormalizedLandmark) this.instance).getY();
            }

            public Builder setY(float value) {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).setY(value);
                return this;
            }

            public Builder clearY() {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).clearY();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public boolean hasZ() {
                return ((NormalizedLandmark) this.instance).hasZ();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public float getZ() {
                return ((NormalizedLandmark) this.instance).getZ();
            }

            public Builder setZ(float value) {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).setZ(value);
                return this;
            }

            public Builder clearZ() {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).clearZ();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public boolean hasVisibility() {
                return ((NormalizedLandmark) this.instance).hasVisibility();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public float getVisibility() {
                return ((NormalizedLandmark) this.instance).getVisibility();
            }

            public Builder setVisibility(float value) {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).setVisibility(value);
                return this;
            }

            public Builder clearVisibility() {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).clearVisibility();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public boolean hasPresence() {
                return ((NormalizedLandmark) this.instance).hasPresence();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkOrBuilder
            public float getPresence() {
                return ((NormalizedLandmark) this.instance).getPresence();
            }

            public Builder setPresence(float value) {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).setPresence(value);
                return this;
            }

            public Builder clearPresence() {
                copyOnWrite();
                ((NormalizedLandmark) this.instance).clearPresence();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new NormalizedLandmark();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "x_", "y_", "z_", "visibility_", "presence_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0005��\u0001\u0001\u0005\u0005������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003\u0005\u0001\u0004", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<NormalizedLandmark> parser = PARSER;
                    if (parser == null) {
                        synchronized (NormalizedLandmark.class) {
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
            NormalizedLandmark defaultInstance = new NormalizedLandmark();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(NormalizedLandmark.class, defaultInstance);
        }

        public static NormalizedLandmark getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<NormalizedLandmark> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$NormalizedLandmarkList.class */
    public static final class NormalizedLandmarkList extends GeneratedMessageLite<NormalizedLandmarkList, Builder> implements NormalizedLandmarkListOrBuilder {
        public static final int LANDMARK_FIELD_NUMBER = 1;
        private Internal.ProtobufList<NormalizedLandmark> landmark_ = emptyProtobufList();
        private static final NormalizedLandmarkList DEFAULT_INSTANCE;
        private static volatile Parser<NormalizedLandmarkList> PARSER;

        private NormalizedLandmarkList() {
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkListOrBuilder
        public List<NormalizedLandmark> getLandmarkList() {
            return this.landmark_;
        }

        public List<? extends NormalizedLandmarkOrBuilder> getLandmarkOrBuilderList() {
            return this.landmark_;
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkListOrBuilder
        public int getLandmarkCount() {
            return this.landmark_.size();
        }

        @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkListOrBuilder
        public NormalizedLandmark getLandmark(int index) {
            return this.landmark_.get(index);
        }

        public NormalizedLandmarkOrBuilder getLandmarkOrBuilder(int index) {
            return this.landmark_.get(index);
        }

        private void ensureLandmarkIsMutable() {
            if (!this.landmark_.isModifiable()) {
                this.landmark_ = GeneratedMessageLite.mutableCopy(this.landmark_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setLandmark(int index, NormalizedLandmark value) {
            value.getClass();
            ensureLandmarkIsMutable();
            this.landmark_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addLandmark(NormalizedLandmark value) {
            value.getClass();
            ensureLandmarkIsMutable();
            this.landmark_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addLandmark(int index, NormalizedLandmark value) {
            value.getClass();
            ensureLandmarkIsMutable();
            this.landmark_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllLandmark(Iterable<? extends NormalizedLandmark> values) {
            ensureLandmarkIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.landmark_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearLandmark() {
            this.landmark_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeLandmark(int index) {
            ensureLandmarkIsMutable();
            this.landmark_.remove(index);
        }

        public static NormalizedLandmarkList parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static NormalizedLandmarkList parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static NormalizedLandmarkList parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static NormalizedLandmarkList parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static NormalizedLandmarkList parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static NormalizedLandmarkList parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static NormalizedLandmarkList parseFrom(InputStream input) throws IOException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static NormalizedLandmarkList parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static NormalizedLandmarkList parseDelimitedFrom(InputStream input) throws IOException {
            return (NormalizedLandmarkList) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static NormalizedLandmarkList parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (NormalizedLandmarkList) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static NormalizedLandmarkList parseFrom(CodedInputStream input) throws IOException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static NormalizedLandmarkList parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (NormalizedLandmarkList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(NormalizedLandmarkList prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LandmarkProto$NormalizedLandmarkList$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<NormalizedLandmarkList, Builder> implements NormalizedLandmarkListOrBuilder {
            private Builder() {
                super(NormalizedLandmarkList.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkListOrBuilder
            public List<NormalizedLandmark> getLandmarkList() {
                return Collections.unmodifiableList(((NormalizedLandmarkList) this.instance).getLandmarkList());
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkListOrBuilder
            public int getLandmarkCount() {
                return ((NormalizedLandmarkList) this.instance).getLandmarkCount();
            }

            @Override // com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkListOrBuilder
            public NormalizedLandmark getLandmark(int index) {
                return ((NormalizedLandmarkList) this.instance).getLandmark(index);
            }

            public Builder setLandmark(int index, NormalizedLandmark value) {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).setLandmark(index, value);
                return this;
            }

            public Builder setLandmark(int index, NormalizedLandmark.Builder builderForValue) {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).setLandmark(index, builderForValue.build());
                return this;
            }

            public Builder addLandmark(NormalizedLandmark value) {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).addLandmark(value);
                return this;
            }

            public Builder addLandmark(int index, NormalizedLandmark value) {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).addLandmark(index, value);
                return this;
            }

            public Builder addLandmark(NormalizedLandmark.Builder builderForValue) {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).addLandmark(builderForValue.build());
                return this;
            }

            public Builder addLandmark(int index, NormalizedLandmark.Builder builderForValue) {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).addLandmark(index, builderForValue.build());
                return this;
            }

            public Builder addAllLandmark(Iterable<? extends NormalizedLandmark> values) {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).addAllLandmark(values);
                return this;
            }

            public Builder clearLandmark() {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).clearLandmark();
                return this;
            }

            public Builder removeLandmark(int index) {
                copyOnWrite();
                ((NormalizedLandmarkList) this.instance).removeLandmark(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new NormalizedLandmarkList();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"landmark_", NormalizedLandmark.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<NormalizedLandmarkList> parser = PARSER;
                    if (parser == null) {
                        synchronized (NormalizedLandmarkList.class) {
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
            NormalizedLandmarkList defaultInstance = new NormalizedLandmarkList();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(NormalizedLandmarkList.class, defaultInstance);
        }

        public static NormalizedLandmarkList getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<NormalizedLandmarkList> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}