package com.google.ar.sceneform.animation;

import android.animation.ObjectAnimator;
import android.text.TextUtils;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes.dex */
public interface AnimatableModel {
    default ObjectAnimator animate(boolean z) {
        ObjectAnimator ofAllAnimations = ModelAnimator.ofAllAnimations(this);
        if (z) {
            ofAllAnimations.setRepeatCount(-1);
        } else {
            ofAllAnimations.setRepeatCount(0);
        }
        return ofAllAnimations;
    }

    boolean applyAnimationChange(ModelAnimation modelAnimation);

    ModelAnimation getAnimation(int i);

    default ModelAnimation getAnimation(String str) {
        int animationIndex = getAnimationIndex(str);
        if (animationIndex != -1) {
            return getAnimation(animationIndex);
        }
        return null;
    }

    int getAnimationCount();

    default int getAnimationIndex(String str) {
        for (int i = 0; i < getAnimationCount(); i++) {
            if (TextUtils.equals(getAnimation(i).getName(), str)) {
                return i;
            }
        }
        return -1;
    }

    default String getAnimationName(int i) {
        return getAnimation(i).getName();
    }

    default List<String> getAnimationNames() {
        ArrayList arrayList = new ArrayList();
        for (int i = 0; i < getAnimationCount(); i++) {
            arrayList.add(getAnimation(i).getName());
        }
        return arrayList;
    }

    default ModelAnimation getAnimationOrThrow(String str) {
        return (ModelAnimation) Preconditions.checkNotNull(getAnimation(str), "No animation found with the given name");
    }

    default boolean hasAnimations() {
        return getAnimationCount() > 0;
    }

    default void onModelAnimationChanged(ModelAnimation modelAnimation) {
        if (applyAnimationChange(modelAnimation)) {
            modelAnimation.setDirty(false);
        }
    }

    default void setAnimationsFramePosition(int i) {
        for (int i2 = 0; i2 < getAnimationCount(); i2++) {
            getAnimation(i2).setFramePosition(i);
        }
    }

    default void setAnimationsTimePosition(float f2) {
        for (int i = 0; i < getAnimationCount(); i++) {
            getAnimation(i).setTimePosition(f2);
        }
    }

    default ObjectAnimator animate(String... strArr) {
        return ModelAnimator.ofAnimation(this, strArr);
    }

    default ObjectAnimator animate(int... iArr) {
        return ModelAnimator.ofAnimation(this, iArr);
    }

    default ObjectAnimator animate(ModelAnimation... modelAnimationArr) {
        return ModelAnimator.ofAnimation(this, modelAnimationArr);
    }
}