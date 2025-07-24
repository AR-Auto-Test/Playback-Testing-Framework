package com.google.ar.sceneform.animation;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ObjectAnimator;
import android.text.TextUtils;
import android.util.Property;
import android.view.animation.LinearInterpolator;
import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes.dex */
public class ModelAnimator {

    /* loaded from: classes.dex */
    public static class PropertyValuesHolder {

        /* loaded from: classes.dex */
        public static class AnimationProperty<T> extends Property<AnimatableModel, T> {
            public WeakReference<ModelAnimation> animation;
            public String animationName;
            public Property<ModelAnimation, T> property;

            /* JADX WARN: Illegal instructions before constructor call */
            /*
                Code decompiled incorrectly, please refer to instructions dump.
            */
            public AnimationProperty(ModelAnimation modelAnimation, Property<ModelAnimation, T> property) {
                super(r0, r1.toString());
                Class<T> type = property.getType();
                StringBuilder x = a.x("animation[");
                x.append(modelAnimation.getName());
                x.append("].");
                x.append(property.getName());
                this.animationName = null;
                this.property = property;
                this.animation = new WeakReference<>(modelAnimation);
            }

            private ModelAnimation getAnimation(AnimatableModel animatableModel) {
                WeakReference<ModelAnimation> weakReference = this.animation;
                if (weakReference == null && weakReference.get() == null) {
                    this.animation = new WeakReference<>(ModelAnimator.getAnimationByName(animatableModel, this.animationName));
                }
                return this.animation.get();
            }

            /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
            /* JADX DEBUG: Multi-variable search result rejected for r2v0, resolved type: java.lang.Object */
            /* JADX WARN: Multi-variable type inference failed */
            @Override // android.util.Property
            public /* bridge */ /* synthetic */ void set(AnimatableModel animatableModel, Object obj) {
                set2(animatableModel, (AnimatableModel) obj);
            }

            /* JADX DEBUG: Method merged with bridge method */
            @Override // android.util.Property
            public T get(AnimatableModel animatableModel) {
                return this.property.get(getAnimation(animatableModel));
            }

            /* renamed from: set  reason: avoid collision after fix types in other method */
            public void set2(AnimatableModel animatableModel, T t) {
                this.property.set(getAnimation(animatableModel), t);
            }

            /* JADX WARN: Illegal instructions before constructor call */
            /*
                Code decompiled incorrectly, please refer to instructions dump.
            */
            public AnimationProperty(String str, Property<ModelAnimation, T> property) {
                super(r0, r1.toString());
                Class<T> type = property.getType();
                StringBuilder B = a.B("animation[", str, "].");
                B.append(property.getName());
                this.animationName = null;
                this.property = property;
                this.animationName = str;
            }
        }

        public static android.animation.PropertyValuesHolder ofAnimation(ModelAnimation modelAnimation) {
            return ofAnimationTime(modelAnimation, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, modelAnimation.getDuration());
        }

        public static android.animation.PropertyValuesHolder ofAnimationFraction(String str, float... fArr) {
            return android.animation.PropertyValuesHolder.ofFloat(new AnimationProperty(str, ModelAnimation.FRACTION_POSITION), fArr);
        }

        public static android.animation.PropertyValuesHolder ofAnimationFrame(String str, int... iArr) {
            return android.animation.PropertyValuesHolder.ofInt(new AnimationProperty(str, ModelAnimation.FRAME_POSITION), iArr);
        }

        public static android.animation.PropertyValuesHolder ofAnimationTime(String str, float... fArr) {
            return android.animation.PropertyValuesHolder.ofFloat(new AnimationProperty(str, ModelAnimation.TIME_POSITION), fArr);
        }

        public static android.animation.PropertyValuesHolder ofAnimationFraction(ModelAnimation modelAnimation, float... fArr) {
            return android.animation.PropertyValuesHolder.ofFloat(new AnimationProperty(modelAnimation, ModelAnimation.FRACTION_POSITION), fArr);
        }

        public static android.animation.PropertyValuesHolder ofAnimationFrame(ModelAnimation modelAnimation, int... iArr) {
            return android.animation.PropertyValuesHolder.ofInt(new AnimationProperty(modelAnimation, ModelAnimation.FRAME_POSITION), iArr);
        }

        public static android.animation.PropertyValuesHolder ofAnimationTime(ModelAnimation modelAnimation, float... fArr) {
            return android.animation.PropertyValuesHolder.ofFloat(new AnimationProperty(modelAnimation, ModelAnimation.TIME_POSITION), fArr);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static ModelAnimation getAnimationByName(AnimatableModel animatableModel, String str) {
        for (int i = 0; i < animatableModel.getAnimationCount(); i++) {
            if (TextUtils.equals(animatableModel.getAnimation(i).getName(), str)) {
                return animatableModel.getAnimation(i);
            }
        }
        return null;
    }

    public static ObjectAnimator ofAllAnimations(AnimatableModel animatableModel) {
        int animationCount = animatableModel.getAnimationCount();
        ModelAnimation[] modelAnimationArr = new ModelAnimation[animationCount];
        for (int i = 0; i < animationCount; i++) {
            modelAnimationArr[i] = animatableModel.getAnimation(i);
        }
        return ofAnimation(animatableModel, modelAnimationArr);
    }

    public static ObjectAnimator ofAnimation(AnimatableModel animatableModel, String... strArr) {
        ModelAnimation[] modelAnimationArr = new ModelAnimation[strArr.length];
        for (int i = 0; i < strArr.length; i++) {
            modelAnimationArr[i] = getAnimationByName(animatableModel, strArr[i]);
        }
        return ofAnimation(animatableModel, modelAnimationArr);
    }

    public static ObjectAnimator ofAnimationFraction(AnimatableModel animatableModel, String str, float... fArr) {
        return ofAnimationFraction(animatableModel, getAnimationByName(animatableModel, str), fArr);
    }

    public static ObjectAnimator ofAnimationFrame(AnimatableModel animatableModel, String str, int... iArr) {
        return ofAnimationFrame(animatableModel, getAnimationByName(animatableModel, str), iArr);
    }

    public static ObjectAnimator ofAnimationTime(AnimatableModel animatableModel, String str, float... fArr) {
        return ofAnimationTime(animatableModel, getAnimationByName(animatableModel, str), fArr);
    }

    public static List<ObjectAnimator> ofMultipleAnimations(AnimatableModel animatableModel, String... strArr) {
        ArrayList arrayList = new ArrayList();
        for (int i = 0; i < strArr.length; i++) {
            arrayList.add(ofAnimation(animatableModel, strArr[i]));
        }
        return arrayList;
    }

    private static ObjectAnimator ofPropertyValuesHolder(AnimatableModel animatableModel, final ModelAnimation modelAnimation, android.animation.PropertyValuesHolder propertyValuesHolder) {
        ObjectAnimator ofPropertyValuesHolder = ofPropertyValuesHolder(animatableModel, propertyValuesHolder);
        ofPropertyValuesHolder.addListener(new AnimatorListenerAdapter() { // from class: com.google.ar.sceneform.animation.ModelAnimator.2
            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
                super.onAnimationCancel(animator);
                ModelAnimation.this.setTimePosition(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            }
        });
        ofPropertyValuesHolder.setDuration(modelAnimation.getDurationMillis());
        ofPropertyValuesHolder.setAutoCancel(true);
        return ofPropertyValuesHolder;
    }

    public static ObjectAnimator ofAnimationFraction(AnimatableModel animatableModel, int i, float... fArr) {
        return ofAnimationFraction(animatableModel, animatableModel.getAnimation(i), fArr);
    }

    public static ObjectAnimator ofAnimationFrame(AnimatableModel animatableModel, int i, int... iArr) {
        return ofAnimationFrame(animatableModel, animatableModel.getAnimation(i), iArr);
    }

    public static ObjectAnimator ofAnimationTime(AnimatableModel animatableModel, int i, float... fArr) {
        return ofAnimationTime(animatableModel, animatableModel.getAnimation(i), fArr);
    }

    public static ObjectAnimator ofAnimationFraction(AnimatableModel animatableModel, ModelAnimation modelAnimation, float... fArr) {
        return ofPropertyValuesHolder(animatableModel, modelAnimation, PropertyValuesHolder.ofAnimationFraction(modelAnimation, fArr));
    }

    public static ObjectAnimator ofAnimationFrame(AnimatableModel animatableModel, ModelAnimation modelAnimation, int... iArr) {
        return ofPropertyValuesHolder(animatableModel, modelAnimation, PropertyValuesHolder.ofAnimationFrame(modelAnimation, iArr));
    }

    public static ObjectAnimator ofAnimationTime(AnimatableModel animatableModel, ModelAnimation modelAnimation, float... fArr) {
        return ofPropertyValuesHolder(animatableModel, modelAnimation, PropertyValuesHolder.ofAnimationTime(modelAnimation, fArr));
    }

    public static ObjectAnimator ofAnimation(AnimatableModel animatableModel, int... iArr) {
        ModelAnimation[] modelAnimationArr = new ModelAnimation[iArr.length];
        for (int i = 0; i < iArr.length; i++) {
            modelAnimationArr[i] = animatableModel.getAnimation(iArr[i]);
        }
        return ofAnimation(animatableModel, modelAnimationArr);
    }

    public static ObjectAnimator ofPropertyValuesHolder(AnimatableModel animatableModel, android.animation.PropertyValuesHolder... propertyValuesHolderArr) {
        ObjectAnimator ofPropertyValuesHolder = ObjectAnimator.ofPropertyValuesHolder(animatableModel, propertyValuesHolderArr);
        ofPropertyValuesHolder.setInterpolator(new LinearInterpolator());
        ofPropertyValuesHolder.setRepeatCount(-1);
        return ofPropertyValuesHolder;
    }

    public static ObjectAnimator ofAnimation(AnimatableModel animatableModel, final ModelAnimation... modelAnimationArr) {
        android.animation.PropertyValuesHolder[] propertyValuesHolderArr = new android.animation.PropertyValuesHolder[modelAnimationArr.length];
        long j = 0;
        for (int i = 0; i < modelAnimationArr.length; i++) {
            j = Math.max(j, modelAnimationArr[i].getDurationMillis());
            propertyValuesHolderArr[i] = PropertyValuesHolder.ofAnimationTime(modelAnimationArr[i], StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, modelAnimationArr[i].getDuration());
        }
        ObjectAnimator ofPropertyValuesHolder = ofPropertyValuesHolder(animatableModel, propertyValuesHolderArr);
        ofPropertyValuesHolder.setDuration(j);
        ofPropertyValuesHolder.addListener(new AnimatorListenerAdapter() { // from class: com.google.ar.sceneform.animation.ModelAnimator.1
            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
                super.onAnimationCancel(animator);
                for (ModelAnimation modelAnimation : modelAnimationArr) {
                    modelAnimation.setTimePosition(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                }
            }
        });
        ofPropertyValuesHolder.setAutoCancel(true);
        return ofPropertyValuesHolder;
    }
}