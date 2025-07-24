package com.google.mediapipe.graphs.instantmotiontracking;

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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/graphs/instantmotiontracking/StickerBufferProto.class */
public final class StickerBufferProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/graphs/instantmotiontracking/StickerBufferProto$StickerOrBuilder.class */
    public interface StickerOrBuilder extends MessageLiteOrBuilder {
        boolean hasId();

        int getId();

        boolean hasX();

        float getX();

        boolean hasY();

        float getY();

        boolean hasRotation();

        float getRotation();

        boolean hasScale();

        float getScale();

        boolean hasRenderId();

        int getRenderId();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/graphs/instantmotiontracking/StickerBufferProto$StickerRollOrBuilder.class */
    public interface StickerRollOrBuilder extends MessageLiteOrBuilder {
        List<Sticker> getStickerList();

        Sticker getSticker(int index);

        int getStickerCount();
    }

    private StickerBufferProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/graphs/instantmotiontracking/StickerBufferProto$Sticker.class */
    public static final class Sticker extends GeneratedMessageLite<Sticker, Builder> implements StickerOrBuilder {
        private int bitField0_;
        public static final int ID_FIELD_NUMBER = 1;
        private int id_;
        public static final int X_FIELD_NUMBER = 2;
        private float x_;
        public static final int Y_FIELD_NUMBER = 3;
        private float y_;
        public static final int ROTATION_FIELD_NUMBER = 4;
        private float rotation_;
        public static final int SCALE_FIELD_NUMBER = 5;
        private float scale_;
        public static final int RENDER_ID_FIELD_NUMBER = 6;
        private int renderId_;
        private static final Sticker DEFAULT_INSTANCE;
        private static volatile Parser<Sticker> PARSER;

        private Sticker() {
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public boolean hasId() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public int getId() {
            return this.id_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setId(int value) {
            this.bitField0_ |= 1;
            this.id_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearId() {
            this.bitField0_ &= -2;
            this.id_ = 0;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public boolean hasX() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public float getX() {
            return this.x_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setX(float value) {
            this.bitField0_ |= 2;
            this.x_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearX() {
            this.bitField0_ &= -3;
            this.x_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public boolean hasY() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public float getY() {
            return this.y_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setY(float value) {
            this.bitField0_ |= 4;
            this.y_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearY() {
            this.bitField0_ &= -5;
            this.y_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public boolean hasRotation() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public float getRotation() {
            return this.rotation_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRotation(float value) {
            this.bitField0_ |= 8;
            this.rotation_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRotation() {
            this.bitField0_ &= -9;
            this.rotation_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public boolean hasScale() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public float getScale() {
            return this.scale_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setScale(float value) {
            this.bitField0_ |= 16;
            this.scale_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearScale() {
            this.bitField0_ &= -17;
            this.scale_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public boolean hasRenderId() {
            return (this.bitField0_ & 32) != 0;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
        public int getRenderId() {
            return this.renderId_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRenderId(int value) {
            this.bitField0_ |= 32;
            this.renderId_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRenderId() {
            this.bitField0_ &= -33;
            this.renderId_ = 0;
        }

        public static Sticker parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Sticker parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Sticker parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Sticker parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Sticker parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Sticker parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Sticker parseFrom(InputStream input) throws IOException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static Sticker parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Sticker parseDelimitedFrom(InputStream input) throws IOException {
            return (Sticker) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static Sticker parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Sticker) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Sticker parseFrom(CodedInputStream input) throws IOException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static Sticker parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Sticker) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(Sticker prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/graphs/instantmotiontracking/StickerBufferProto$Sticker$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<Sticker, Builder> implements StickerOrBuilder {
            private Builder() {
                super(Sticker.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public boolean hasId() {
                return ((Sticker) this.instance).hasId();
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public int getId() {
                return ((Sticker) this.instance).getId();
            }

            public Builder setId(int value) {
                copyOnWrite();
                ((Sticker) this.instance).setId(value);
                return this;
            }

            public Builder clearId() {
                copyOnWrite();
                ((Sticker) this.instance).clearId();
                return this;
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public boolean hasX() {
                return ((Sticker) this.instance).hasX();
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public float getX() {
                return ((Sticker) this.instance).getX();
            }

            public Builder setX(float value) {
                copyOnWrite();
                ((Sticker) this.instance).setX(value);
                return this;
            }

            public Builder clearX() {
                copyOnWrite();
                ((Sticker) this.instance).clearX();
                return this;
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public boolean hasY() {
                return ((Sticker) this.instance).hasY();
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public float getY() {
                return ((Sticker) this.instance).getY();
            }

            public Builder setY(float value) {
                copyOnWrite();
                ((Sticker) this.instance).setY(value);
                return this;
            }

            public Builder clearY() {
                copyOnWrite();
                ((Sticker) this.instance).clearY();
                return this;
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public boolean hasRotation() {
                return ((Sticker) this.instance).hasRotation();
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public float getRotation() {
                return ((Sticker) this.instance).getRotation();
            }

            public Builder setRotation(float value) {
                copyOnWrite();
                ((Sticker) this.instance).setRotation(value);
                return this;
            }

            public Builder clearRotation() {
                copyOnWrite();
                ((Sticker) this.instance).clearRotation();
                return this;
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public boolean hasScale() {
                return ((Sticker) this.instance).hasScale();
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public float getScale() {
                return ((Sticker) this.instance).getScale();
            }

            public Builder setScale(float value) {
                copyOnWrite();
                ((Sticker) this.instance).setScale(value);
                return this;
            }

            public Builder clearScale() {
                copyOnWrite();
                ((Sticker) this.instance).clearScale();
                return this;
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public boolean hasRenderId() {
                return ((Sticker) this.instance).hasRenderId();
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerOrBuilder
            public int getRenderId() {
                return ((Sticker) this.instance).getRenderId();
            }

            public Builder setRenderId(int value) {
                copyOnWrite();
                ((Sticker) this.instance).setRenderId(value);
                return this;
            }

            public Builder clearRenderId() {
                copyOnWrite();
                ((Sticker) this.instance).clearRenderId();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new Sticker();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "id_", "x_", "y_", "rotation_", "scale_", "renderId_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0006��\u0001\u0001\u0006\u0006������\u0001\u0004��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003\u0005\u0001\u0004\u0006\u0004\u0005", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<Sticker> parser = PARSER;
                    if (parser == null) {
                        synchronized (Sticker.class) {
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
            Sticker defaultInstance = new Sticker();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(Sticker.class, defaultInstance);
        }

        public static Sticker getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<Sticker> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/graphs/instantmotiontracking/StickerBufferProto$StickerRoll.class */
    public static final class StickerRoll extends GeneratedMessageLite<StickerRoll, Builder> implements StickerRollOrBuilder {
        public static final int STICKER_FIELD_NUMBER = 1;
        private Internal.ProtobufList<Sticker> sticker_ = emptyProtobufList();
        private static final StickerRoll DEFAULT_INSTANCE;
        private static volatile Parser<StickerRoll> PARSER;

        private StickerRoll() {
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerRollOrBuilder
        public List<Sticker> getStickerList() {
            return this.sticker_;
        }

        public List<? extends StickerOrBuilder> getStickerOrBuilderList() {
            return this.sticker_;
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerRollOrBuilder
        public int getStickerCount() {
            return this.sticker_.size();
        }

        @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerRollOrBuilder
        public Sticker getSticker(int index) {
            return this.sticker_.get(index);
        }

        public StickerOrBuilder getStickerOrBuilder(int index) {
            return this.sticker_.get(index);
        }

        private void ensureStickerIsMutable() {
            if (!this.sticker_.isModifiable()) {
                this.sticker_ = GeneratedMessageLite.mutableCopy(this.sticker_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setSticker(int index, Sticker value) {
            value.getClass();
            ensureStickerIsMutable();
            this.sticker_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addSticker(Sticker value) {
            value.getClass();
            ensureStickerIsMutable();
            this.sticker_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addSticker(int index, Sticker value) {
            value.getClass();
            ensureStickerIsMutable();
            this.sticker_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllSticker(Iterable<? extends Sticker> values) {
            ensureStickerIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.sticker_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearSticker() {
            this.sticker_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeSticker(int index) {
            ensureStickerIsMutable();
            this.sticker_.remove(index);
        }

        public static StickerRoll parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static StickerRoll parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static StickerRoll parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static StickerRoll parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static StickerRoll parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static StickerRoll parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static StickerRoll parseFrom(InputStream input) throws IOException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static StickerRoll parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static StickerRoll parseDelimitedFrom(InputStream input) throws IOException {
            return (StickerRoll) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static StickerRoll parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (StickerRoll) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static StickerRoll parseFrom(CodedInputStream input) throws IOException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static StickerRoll parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (StickerRoll) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(StickerRoll prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/graphs/instantmotiontracking/StickerBufferProto$StickerRoll$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<StickerRoll, Builder> implements StickerRollOrBuilder {
            private Builder() {
                super(StickerRoll.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerRollOrBuilder
            public List<Sticker> getStickerList() {
                return Collections.unmodifiableList(((StickerRoll) this.instance).getStickerList());
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerRollOrBuilder
            public int getStickerCount() {
                return ((StickerRoll) this.instance).getStickerCount();
            }

            @Override // com.google.mediapipe.graphs.instantmotiontracking.StickerBufferProto.StickerRollOrBuilder
            public Sticker getSticker(int index) {
                return ((StickerRoll) this.instance).getSticker(index);
            }

            public Builder setSticker(int index, Sticker value) {
                copyOnWrite();
                ((StickerRoll) this.instance).setSticker(index, value);
                return this;
            }

            public Builder setSticker(int index, Sticker.Builder builderForValue) {
                copyOnWrite();
                ((StickerRoll) this.instance).setSticker(index, builderForValue.build());
                return this;
            }

            public Builder addSticker(Sticker value) {
                copyOnWrite();
                ((StickerRoll) this.instance).addSticker(value);
                return this;
            }

            public Builder addSticker(int index, Sticker value) {
                copyOnWrite();
                ((StickerRoll) this.instance).addSticker(index, value);
                return this;
            }

            public Builder addSticker(Sticker.Builder builderForValue) {
                copyOnWrite();
                ((StickerRoll) this.instance).addSticker(builderForValue.build());
                return this;
            }

            public Builder addSticker(int index, Sticker.Builder builderForValue) {
                copyOnWrite();
                ((StickerRoll) this.instance).addSticker(index, builderForValue.build());
                return this;
            }

            public Builder addAllSticker(Iterable<? extends Sticker> values) {
                copyOnWrite();
                ((StickerRoll) this.instance).addAllSticker(values);
                return this;
            }

            public Builder clearSticker() {
                copyOnWrite();
                ((StickerRoll) this.instance).clearSticker();
                return this;
            }

            public Builder removeSticker(int index) {
                copyOnWrite();
                ((StickerRoll) this.instance).removeSticker(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new StickerRoll();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"sticker_", Sticker.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<StickerRoll> parser = PARSER;
                    if (parser == null) {
                        synchronized (StickerRoll.class) {
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
            StickerRoll defaultInstance = new StickerRoll();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(StickerRoll.class, defaultInstance);
        }

        public static StickerRoll getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<StickerRoll> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}