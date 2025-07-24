package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.TypedArray;
import android.graphics.Color;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;

/* compiled from: ThemeUtils.java */
/* loaded from: classes.dex */
public class t0 {

    /* renamed from: a  reason: collision with root package name */
    public static final ThreadLocal<TypedValue> f921a = new ThreadLocal<>();

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f922b = {-16842910};

    /* renamed from: c  reason: collision with root package name */
    public static final int[] f923c = {16842908};

    /* renamed from: d  reason: collision with root package name */
    public static final int[] f924d = {16842919};

    /* renamed from: e  reason: collision with root package name */
    public static final int[] f925e = {16842912};

    /* renamed from: f  reason: collision with root package name */
    public static final int[] f926f = new int[0];

    /* renamed from: g  reason: collision with root package name */
    public static final int[] f927g = new int[1];

    public static void a(View view, Context context) {
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(b.b.b.j);
        try {
            if (!obtainStyledAttributes.hasValue(115)) {
                Log.e("ThemeUtils", "View " + view.getClass() + " is an AppCompat widget that can only be used with a Theme.AppCompat theme (or descendant).");
            }
        } finally {
            obtainStyledAttributes.recycle();
        }
    }

    public static int b(Context context, int i) {
        ColorStateList d2 = d(context, i);
        if (d2 != null && d2.isStateful()) {
            return d2.getColorForState(f922b, d2.getDefaultColor());
        }
        ThreadLocal<TypedValue> threadLocal = f921a;
        TypedValue typedValue = threadLocal.get();
        if (typedValue == null) {
            typedValue = new TypedValue();
            threadLocal.set(typedValue);
        }
        context.getTheme().resolveAttribute(16842803, typedValue, true);
        float f2 = typedValue.getFloat();
        int c2 = c(context, i);
        return b.j.d.a.c(c2, Math.round(Color.alpha(c2) * f2));
    }

    public static int c(Context context, int i) {
        int[] iArr = f927g;
        iArr[0] = i;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes((AttributeSet) null, iArr);
        try {
            return obtainStyledAttributes.getColor(0, 0);
        } finally {
            obtainStyledAttributes.recycle();
        }
    }

    public static ColorStateList d(Context context, int i) {
        ColorStateList colorStateList;
        int resourceId;
        int[] iArr = f927g;
        iArr[0] = i;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes((AttributeSet) null, iArr);
        try {
            if (obtainStyledAttributes.hasValue(0) && (resourceId = obtainStyledAttributes.getResourceId(0, 0)) != 0) {
                ThreadLocal<TypedValue> threadLocal = b.b.d.a.a.f630a;
                colorStateList = context.getColorStateList(resourceId);
                if (colorStateList != null) {
                    return colorStateList;
                }
            }
            colorStateList = obtainStyledAttributes.getColorStateList(0);
            return colorStateList;
        } finally {
            obtainStyledAttributes.recycle();
        }
    }
}