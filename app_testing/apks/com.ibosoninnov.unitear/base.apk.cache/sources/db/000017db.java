package com.google.android.material.internal;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import b.z.j;
import b.z.p;
import java.util.Map;

/* loaded from: classes.dex */
public class TextScale extends j {
    private static final String PROPNAME_SCALE = "android:textscale:scale";

    private void captureValues(p pVar) {
        View view = pVar.f2914b;
        if (view instanceof TextView) {
            pVar.f2913a.put(PROPNAME_SCALE, Float.valueOf(((TextView) view).getScaleX()));
        }
    }

    @Override // b.z.j
    public void captureEndValues(p pVar) {
        captureValues(pVar);
    }

    @Override // b.z.j
    public void captureStartValues(p pVar) {
        captureValues(pVar);
    }

    @Override // b.z.j
    public Animator createAnimator(ViewGroup viewGroup, p pVar, p pVar2) {
        if (pVar == null || pVar2 == null || !(pVar.f2914b instanceof TextView)) {
            return null;
        }
        View view = pVar2.f2914b;
        if (view instanceof TextView) {
            final TextView textView = (TextView) view;
            Map<String, Object> map = pVar.f2913a;
            Map<String, Object> map2 = pVar2.f2913a;
            float floatValue = map.get(PROPNAME_SCALE) != null ? ((Float) map.get(PROPNAME_SCALE)).floatValue() : 1.0f;
            float floatValue2 = map2.get(PROPNAME_SCALE) != null ? ((Float) map2.get(PROPNAME_SCALE)).floatValue() : 1.0f;
            if (floatValue == floatValue2) {
                return null;
            }
            ValueAnimator ofFloat = ValueAnimator.ofFloat(floatValue, floatValue2);
            ofFloat.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.google.android.material.internal.TextScale.1
                @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                public void onAnimationUpdate(ValueAnimator valueAnimator) {
                    float floatValue3 = ((Float) valueAnimator.getAnimatedValue()).floatValue();
                    textView.setScaleX(floatValue3);
                    textView.setScaleY(floatValue3);
                }
            });
            return ofFloat;
        }
        return null;
    }
}