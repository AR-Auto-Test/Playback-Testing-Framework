package b.b.h;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.RectF;
import android.os.Build;
import android.text.Layout;
import android.text.StaticLayout;
import android.text.TextDirectionHeuristic;
import android.text.TextDirectionHeuristics;
import android.text.TextPaint;
import android.text.method.TransformationMethod;
import android.util.Log;
import android.util.TypedValue;
import android.widget.TextView;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.ConcurrentHashMap;

/* compiled from: AppCompatTextViewAutoSizeHelper.java */
/* loaded from: classes.dex */
public class a0 {

    /* renamed from: a  reason: collision with root package name */
    public static final RectF f779a = new RectF();

    /* renamed from: b  reason: collision with root package name */
    public static ConcurrentHashMap<String, Method> f780b = new ConcurrentHashMap<>();

    /* renamed from: c  reason: collision with root package name */
    public static ConcurrentHashMap<String, Field> f781c = new ConcurrentHashMap<>();

    /* renamed from: d  reason: collision with root package name */
    public int f782d = 0;

    /* renamed from: e  reason: collision with root package name */
    public boolean f783e = false;

    /* renamed from: f  reason: collision with root package name */
    public float f784f = -1.0f;

    /* renamed from: g  reason: collision with root package name */
    public float f785g = -1.0f;

    /* renamed from: h  reason: collision with root package name */
    public float f786h = -1.0f;
    public int[] i = new int[0];
    public boolean j = false;
    public TextPaint k;
    public final TextView l;
    public final Context m;
    public final c n;

    /* compiled from: AppCompatTextViewAutoSizeHelper.java */
    /* loaded from: classes.dex */
    public static class a extends c {
        @Override // b.b.h.a0.c
        public void a(StaticLayout.Builder builder, TextView textView) {
            builder.setTextDirection((TextDirectionHeuristic) a0.e(textView, "getTextDirectionHeuristic", TextDirectionHeuristics.FIRSTSTRONG_LTR));
        }
    }

    /* compiled from: AppCompatTextViewAutoSizeHelper.java */
    /* loaded from: classes.dex */
    public static class b extends a {
        @Override // b.b.h.a0.a, b.b.h.a0.c
        public void a(StaticLayout.Builder builder, TextView textView) {
            builder.setTextDirection(textView.getTextDirectionHeuristic());
        }

        @Override // b.b.h.a0.c
        public boolean b(TextView textView) {
            return textView.isHorizontallyScrollable();
        }
    }

    /* compiled from: AppCompatTextViewAutoSizeHelper.java */
    /* loaded from: classes.dex */
    public static class c {
        public void a(StaticLayout.Builder builder, TextView textView) {
            throw null;
        }

        public boolean b(TextView textView) {
            return ((Boolean) a0.e(textView, "getHorizontallyScrolling", Boolean.FALSE)).booleanValue();
        }
    }

    public a0(TextView textView) {
        this.l = textView;
        this.m = textView.getContext();
        if (Build.VERSION.SDK_INT >= 29) {
            this.n = new b();
        } else {
            this.n = new a();
        }
    }

    public static Method d(String str) {
        try {
            Method method = f780b.get(str);
            if (method == null && (method = TextView.class.getDeclaredMethod(str, new Class[0])) != null) {
                method.setAccessible(true);
                f780b.put(str, method);
            }
            return method;
        } catch (Exception e2) {
            Log.w("ACTVAutoSizeHelper", "Failed to retrieve TextView#" + str + "() method", e2);
            return null;
        }
    }

    public static <T> T e(Object obj, String str, T t) {
        try {
            return (T) d(str).invoke(obj, new Object[0]);
        } catch (Exception e2) {
            Log.w("ACTVAutoSizeHelper", "Failed to invoke TextView#" + str + "() method", e2);
            return t;
        }
    }

    public void a() {
        if (i() && this.f782d != 0) {
            if (this.f783e) {
                if (this.l.getMeasuredHeight() <= 0 || this.l.getMeasuredWidth() <= 0) {
                    return;
                }
                int measuredWidth = this.n.b(this.l) ? 1048576 : (this.l.getMeasuredWidth() - this.l.getTotalPaddingLeft()) - this.l.getTotalPaddingRight();
                int height = (this.l.getHeight() - this.l.getCompoundPaddingBottom()) - this.l.getCompoundPaddingTop();
                if (measuredWidth <= 0 || height <= 0) {
                    return;
                }
                RectF rectF = f779a;
                synchronized (rectF) {
                    rectF.setEmpty();
                    rectF.right = measuredWidth;
                    rectF.bottom = height;
                    float c2 = c(rectF);
                    if (c2 != this.l.getTextSize()) {
                        f(0, c2);
                    }
                }
            }
            this.f783e = true;
        }
    }

    public final int[] b(int[] iArr) {
        int length = iArr.length;
        if (length == 0) {
            return iArr;
        }
        Arrays.sort(iArr);
        ArrayList arrayList = new ArrayList();
        for (int i : iArr) {
            if (i > 0 && Collections.binarySearch(arrayList, Integer.valueOf(i)) < 0) {
                arrayList.add(Integer.valueOf(i));
            }
        }
        if (length == arrayList.size()) {
            return iArr;
        }
        int size = arrayList.size();
        int[] iArr2 = new int[size];
        for (int i2 = 0; i2 < size; i2++) {
            iArr2[i2] = ((Integer) arrayList.get(i2)).intValue();
        }
        return iArr2;
    }

    public final int c(RectF rectF) {
        CharSequence transformation;
        int length = this.i.length;
        if (length != 0) {
            int i = length - 1;
            int i2 = 1;
            int i3 = 0;
            while (i2 <= i) {
                int i4 = (i2 + i) / 2;
                int i5 = this.i[i4];
                CharSequence text = this.l.getText();
                TransformationMethod transformationMethod = this.l.getTransformationMethod();
                if (transformationMethod != null && (transformation = transformationMethod.getTransformation(text, this.l)) != null) {
                    text = transformation;
                }
                int maxLines = this.l.getMaxLines();
                TextPaint textPaint = this.k;
                if (textPaint == null) {
                    this.k = new TextPaint();
                } else {
                    textPaint.reset();
                }
                this.k.set(this.l.getPaint());
                this.k.setTextSize(i5);
                StaticLayout.Builder obtain = StaticLayout.Builder.obtain(text, 0, text.length(), this.k, Math.round(rectF.right));
                obtain.setAlignment((Layout.Alignment) e(this.l, "getLayoutAlignment", Layout.Alignment.ALIGN_NORMAL)).setLineSpacing(this.l.getLineSpacingExtra(), this.l.getLineSpacingMultiplier()).setIncludePad(this.l.getIncludeFontPadding()).setBreakStrategy(this.l.getBreakStrategy()).setHyphenationFrequency(this.l.getHyphenationFrequency()).setMaxLines(maxLines == -1 ? Integer.MAX_VALUE : maxLines);
                try {
                    this.n.a(obtain, this.l);
                } catch (ClassCastException unused) {
                    Log.w("ACTVAutoSizeHelper", "Failed to obtain TextDirectionHeuristic, auto size may be incorrect");
                }
                StaticLayout build = obtain.build();
                if ((maxLines == -1 || (build.getLineCount() <= maxLines && build.getLineEnd(build.getLineCount() - 1) == text.length())) && ((float) build.getHeight()) <= rectF.bottom) {
                    int i6 = i4 + 1;
                    i3 = i2;
                    i2 = i6;
                } else {
                    i3 = i4 - 1;
                    i = i3;
                }
            }
            return this.i[i3];
        }
        throw new IllegalStateException("No available text sizes to choose from.");
    }

    public void f(int i, float f2) {
        Resources resources;
        Context context = this.m;
        if (context == null) {
            resources = Resources.getSystem();
        } else {
            resources = context.getResources();
        }
        float applyDimension = TypedValue.applyDimension(i, f2, resources.getDisplayMetrics());
        if (applyDimension != this.l.getPaint().getTextSize()) {
            this.l.getPaint().setTextSize(applyDimension);
            boolean isInLayout = this.l.isInLayout();
            if (this.l.getLayout() != null) {
                this.f783e = false;
                try {
                    Method d2 = d("nullLayouts");
                    if (d2 != null) {
                        d2.invoke(this.l, new Object[0]);
                    }
                } catch (Exception e2) {
                    Log.w("ACTVAutoSizeHelper", "Failed to invoke TextView#nullLayouts() method", e2);
                }
                if (!isInLayout) {
                    this.l.requestLayout();
                } else {
                    this.l.forceLayout();
                }
                this.l.invalidate();
            }
        }
    }

    public final boolean g() {
        if (i() && this.f782d == 1) {
            if (!this.j || this.i.length == 0) {
                int floor = ((int) Math.floor((this.f786h - this.f785g) / this.f784f)) + 1;
                int[] iArr = new int[floor];
                for (int i = 0; i < floor; i++) {
                    iArr[i] = Math.round((i * this.f784f) + this.f785g);
                }
                this.i = b(iArr);
            }
            this.f783e = true;
        } else {
            this.f783e = false;
        }
        return this.f783e;
    }

    public final boolean h() {
        int[] iArr = this.i;
        int length = iArr.length;
        boolean z = length > 0;
        this.j = z;
        if (z) {
            this.f782d = 1;
            this.f785g = iArr[0];
            this.f786h = iArr[length - 1];
            this.f784f = -1.0f;
        }
        return z;
    }

    public final boolean i() {
        return !(this.l instanceof k);
    }

    public final void j(float f2, float f3, float f4) {
        if (f2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            throw new IllegalArgumentException("Minimum auto-size text size (" + f2 + "px) is less or equal to (0px)");
        } else if (f3 <= f2) {
            throw new IllegalArgumentException("Maximum auto-size text size (" + f3 + "px) is less or equal to minimum auto-size text size (" + f2 + "px)");
        } else if (f4 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            this.f782d = 1;
            this.f785g = f2;
            this.f786h = f3;
            this.f784f = f4;
            this.j = false;
        } else {
            throw new IllegalArgumentException("The auto-size step granularity (" + f4 + "px) is less or equal to (0px)");
        }
    }
}