package b.j.h;

import android.annotation.SuppressLint;
import android.os.Build;
import android.text.PrecomputedText;
import android.text.Spannable;
import android.text.TextDirectionHeuristic;
import android.text.TextPaint;
import android.text.TextUtils;
import android.text.style.MetricAffectingSpan;
import java.util.Objects;

/* compiled from: PrecomputedTextCompat.java */
/* loaded from: classes.dex */
public class b implements Spannable {
    @Override // java.lang.CharSequence
    public char charAt(int i) {
        throw null;
    }

    @Override // android.text.Spanned
    public int getSpanEnd(Object obj) {
        throw null;
    }

    @Override // android.text.Spanned
    public int getSpanFlags(Object obj) {
        throw null;
    }

    @Override // android.text.Spanned
    public int getSpanStart(Object obj) {
        throw null;
    }

    @Override // android.text.Spanned
    @SuppressLint({"NewApi"})
    public <T> T[] getSpans(int i, int i2, Class<T> cls) {
        if (Build.VERSION.SDK_INT >= 29) {
            throw null;
        }
        throw null;
    }

    @Override // java.lang.CharSequence
    public int length() {
        throw null;
    }

    @Override // android.text.Spanned
    public int nextSpanTransition(int i, int i2, Class cls) {
        throw null;
    }

    @Override // android.text.Spannable
    @SuppressLint({"NewApi"})
    public void removeSpan(Object obj) {
        if (!(obj instanceof MetricAffectingSpan)) {
            if (Build.VERSION.SDK_INT >= 29) {
                throw null;
            }
            throw null;
        }
        throw new IllegalArgumentException("MetricAffectingSpan can not be removed from PrecomputedText.");
    }

    @Override // android.text.Spannable
    @SuppressLint({"NewApi"})
    public void setSpan(Object obj, int i, int i2, int i3) {
        if (!(obj instanceof MetricAffectingSpan)) {
            if (Build.VERSION.SDK_INT >= 29) {
                throw null;
            }
            throw null;
        }
        throw new IllegalArgumentException("MetricAffectingSpan can not be set to PrecomputedText.");
    }

    @Override // java.lang.CharSequence
    public CharSequence subSequence(int i, int i2) {
        throw null;
    }

    @Override // java.lang.CharSequence
    public String toString() {
        throw null;
    }

    /* compiled from: PrecomputedTextCompat.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final TextPaint f2178a;

        /* renamed from: b  reason: collision with root package name */
        public final TextDirectionHeuristic f2179b;

        /* renamed from: c  reason: collision with root package name */
        public final int f2180c;

        /* renamed from: d  reason: collision with root package name */
        public final int f2181d;

        @SuppressLint({"NewApi"})
        public a(TextPaint textPaint, TextDirectionHeuristic textDirectionHeuristic, int i, int i2) {
            if (Build.VERSION.SDK_INT >= 29) {
                new PrecomputedText.Params.Builder(textPaint).setBreakStrategy(i).setHyphenationFrequency(i2).setTextDirection(textDirectionHeuristic).build();
            }
            this.f2178a = textPaint;
            this.f2179b = textDirectionHeuristic;
            this.f2180c = i;
            this.f2181d = i2;
        }

        public boolean a(a aVar) {
            if (this.f2180c == aVar.f2180c && this.f2181d == aVar.f2181d && this.f2178a.getTextSize() == aVar.f2178a.getTextSize() && this.f2178a.getTextScaleX() == aVar.f2178a.getTextScaleX() && this.f2178a.getTextSkewX() == aVar.f2178a.getTextSkewX() && this.f2178a.getLetterSpacing() == aVar.f2178a.getLetterSpacing() && TextUtils.equals(this.f2178a.getFontFeatureSettings(), aVar.f2178a.getFontFeatureSettings()) && this.f2178a.getFlags() == aVar.f2178a.getFlags() && this.f2178a.getTextLocales().equals(aVar.f2178a.getTextLocales())) {
                return this.f2178a.getTypeface() == null ? aVar.f2178a.getTypeface() == null : this.f2178a.getTypeface().equals(aVar.f2178a.getTypeface());
            }
            return false;
        }

        public boolean equals(Object obj) {
            if (obj == this) {
                return true;
            }
            if (obj instanceof a) {
                a aVar = (a) obj;
                return a(aVar) && this.f2179b == aVar.f2179b;
            }
            return false;
        }

        public int hashCode() {
            return Objects.hash(Float.valueOf(this.f2178a.getTextSize()), Float.valueOf(this.f2178a.getTextScaleX()), Float.valueOf(this.f2178a.getTextSkewX()), Float.valueOf(this.f2178a.getLetterSpacing()), Integer.valueOf(this.f2178a.getFlags()), this.f2178a.getTextLocales(), this.f2178a.getTypeface(), Boolean.valueOf(this.f2178a.isElegantTextHeight()), this.f2179b, Integer.valueOf(this.f2180c), Integer.valueOf(this.f2181d));
        }

        public String toString() {
            StringBuilder sb = new StringBuilder("{");
            StringBuilder x = c.b.a.a.a.x("textSize=");
            x.append(this.f2178a.getTextSize());
            sb.append(x.toString());
            sb.append(", textScaleX=" + this.f2178a.getTextScaleX());
            sb.append(", textSkewX=" + this.f2178a.getTextSkewX());
            int i = Build.VERSION.SDK_INT;
            StringBuilder x2 = c.b.a.a.a.x(", letterSpacing=");
            x2.append(this.f2178a.getLetterSpacing());
            sb.append(x2.toString());
            sb.append(", elegantTextHeight=" + this.f2178a.isElegantTextHeight());
            sb.append(", textLocale=" + this.f2178a.getTextLocales());
            sb.append(", typeface=" + this.f2178a.getTypeface());
            if (i >= 26) {
                StringBuilder x3 = c.b.a.a.a.x(", variationSettings=");
                x3.append(this.f2178a.getFontVariationSettings());
                sb.append(x3.toString());
            }
            StringBuilder x4 = c.b.a.a.a.x(", textDir=");
            x4.append(this.f2179b);
            sb.append(x4.toString());
            sb.append(", breakStrategy=" + this.f2180c);
            sb.append(", hyphenationFrequency=" + this.f2181d);
            sb.append("}");
            return sb.toString();
        }

        public a(PrecomputedText.Params params) {
            this.f2178a = params.getTextPaint();
            this.f2179b = params.getTextDirection();
            this.f2180c = params.getBreakStrategy();
            this.f2181d = params.getHyphenationFrequency();
            int i = Build.VERSION.SDK_INT;
        }
    }
}