package com.google.ar.sceneform.animation;

import android.text.TextUtils;
import android.util.FloatProperty;
import android.util.IntProperty;
import android.util.Property;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.concurrent.TimeUnit;

/* loaded from: classes.dex */
public class ModelAnimation {
    private float duration;
    private int frameRate;
    private int index;
    private AnimatableModel model;
    private String name;
    public static final FloatProperty<ModelAnimation> TIME_POSITION = new FloatProperty<ModelAnimation>("timePosition") { // from class: com.google.ar.sceneform.animation.ModelAnimation.1
        /* JADX DEBUG: Method merged with bridge method */
        @Override // android.util.Property
        public Float get(ModelAnimation modelAnimation) {
            return Float.valueOf(modelAnimation.getTimePosition());
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // android.util.FloatProperty
        public void setValue(ModelAnimation modelAnimation, float f2) {
            modelAnimation.setTimePosition(f2);
        }
    };
    public static final Property<ModelAnimation, Integer> FRAME_POSITION = new IntProperty<ModelAnimation>("framePosition") { // from class: com.google.ar.sceneform.animation.ModelAnimation.2
        /* JADX DEBUG: Method merged with bridge method */
        @Override // android.util.Property
        public Integer get(ModelAnimation modelAnimation) {
            return Integer.valueOf(modelAnimation.getFramePosition());
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // android.util.IntProperty
        public void setValue(ModelAnimation modelAnimation, int i) {
            modelAnimation.setFramePosition(i);
        }
    };
    public static final Property<ModelAnimation, Float> FRACTION_POSITION = new FloatProperty<ModelAnimation>("fractionPosition") { // from class: com.google.ar.sceneform.animation.ModelAnimation.3
        /* JADX DEBUG: Method merged with bridge method */
        @Override // android.util.Property
        public Float get(ModelAnimation modelAnimation) {
            return Float.valueOf(modelAnimation.getFractionPosition());
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // android.util.FloatProperty
        public void setValue(ModelAnimation modelAnimation, float f2) {
            modelAnimation.setFractionPosition(f2);
        }
    };
    private float timePosition = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    private boolean isDirty = false;

    /* loaded from: classes.dex */
    public static class PropertyValuesHolder {
        public static android.animation.PropertyValuesHolder ofFraction(float... fArr) {
            return android.animation.PropertyValuesHolder.ofFloat(ModelAnimation.FRACTION_POSITION, fArr);
        }

        public static android.animation.PropertyValuesHolder ofFrame(int... iArr) {
            return android.animation.PropertyValuesHolder.ofInt(ModelAnimation.FRAME_POSITION, iArr);
        }

        public static android.animation.PropertyValuesHolder ofTime(float... fArr) {
            return android.animation.PropertyValuesHolder.ofFloat(ModelAnimation.TIME_POSITION, fArr);
        }
    }

    public ModelAnimation(AnimatableModel animatableModel, String str, int i, float f2, int i2) {
        this.model = animatableModel;
        this.index = i;
        this.name = str;
        if (TextUtils.isEmpty(str)) {
            this.name = String.valueOf(i);
        }
        this.frameRate = i2;
        this.duration = f2;
    }

    public static float fractionToTime(float f2, float f3) {
        return f2 * f3;
    }

    public static float frameToTime(int i, int i2) {
        return i / i2;
    }

    public static long secondsToMillis(float f2) {
        return f2 * ((float) TimeUnit.SECONDS.toMillis(1L));
    }

    public static float timeToFraction(float f2, float f3) {
        return f2 / f3;
    }

    public static int timeToFrame(float f2, int i) {
        return (int) (f2 * i);
    }

    public int geIndex() {
        return this.index;
    }

    public float getDuration() {
        return this.duration;
    }

    public long getDurationMillis() {
        return secondsToMillis(getDuration());
    }

    public float getFractionAtTime(float f2) {
        return timeToFraction(f2, getDuration());
    }

    public float getFractionPosition() {
        return getFractionAtTime(getTimePosition());
    }

    public int getFrameAtTime(float f2) {
        return timeToFrame(f2, getFrameRate());
    }

    public int getFrameCount() {
        return timeToFrame(getDuration(), getFrameRate());
    }

    public int getFramePosition() {
        return getFrameAtTime(getTimePosition());
    }

    public int getFrameRate() {
        return this.frameRate;
    }

    public String getName() {
        return this.name;
    }

    public float getTimeAtFraction(float f2) {
        return fractionToTime(f2, getDuration());
    }

    public float getTimeAtFrame(int i) {
        return frameToTime(i, getFrameRate());
    }

    public float getTimePosition() {
        return this.timePosition;
    }

    public boolean isDirty() {
        return this.isDirty;
    }

    public void setDirty(boolean z) {
        this.isDirty = z;
        if (z) {
            this.model.onModelAnimationChanged(this);
        }
    }

    public void setFractionPosition(float f2) {
        setTimePosition(getTimeAtFraction(f2));
    }

    public void setFramePosition(int i) {
        setTimePosition(getTimeAtFrame(i));
    }

    public void setTimePosition(float f2) {
        this.timePosition = f2;
        setDirty(true);
    }
}