package b.z;

import android.animation.TypeEvaluator;
import android.graphics.Rect;

/* compiled from: RectEvaluator.java */
/* loaded from: classes.dex */
public class g implements TypeEvaluator<Rect> {
    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [float, java.lang.Object, java.lang.Object] */
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // android.animation.TypeEvaluator
    public Rect evaluate(float f2, Rect rect, Rect rect2) {
        Rect rect3 = rect;
        Rect rect4 = rect2;
        int i = rect3.left;
        int i2 = i + ((int) ((rect4.left - i) * f2));
        int i3 = rect3.top;
        int i4 = i3 + ((int) ((rect4.top - i3) * f2));
        int i5 = rect3.right;
        int i6 = rect3.bottom;
        return new Rect(i2, i4, i5 + ((int) ((rect4.right - i5) * f2)), i6 + ((int) ((rect4.bottom - i6) * f2)));
    }
}