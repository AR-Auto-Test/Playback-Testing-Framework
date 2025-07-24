package h.a.a;

import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.View;
import android.widget.ImageView;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import pl.droidsonroids.gif.GifInfoHandle;

/* compiled from: GifViewUtils.java */
/* loaded from: classes2.dex */
public final class g {

    /* renamed from: a  reason: collision with root package name */
    public static final List<String> f6245a = Arrays.asList("raw", "drawable", "mipmap");

    public static void a(int i, Drawable drawable) {
        if (drawable instanceof c) {
            GifInfoHandle gifInfoHandle = ((c) drawable).f6232h;
            Objects.requireNonNull(gifInfoHandle);
            if (i >= 0 && i <= 65535) {
                synchronized (gifInfoHandle) {
                    GifInfoHandle.setLoopCount(gifInfoHandle.f6268b, (char) i);
                }
                return;
            }
            throw new IllegalArgumentException("Loop count of range <0, 65535>");
        }
    }

    public static boolean b(ImageView imageView, boolean z, int i) {
        Resources resources = imageView.getResources();
        if (resources != null) {
            try {
                if (f6245a.contains(resources.getResourceTypeName(i))) {
                    c cVar = new c(resources, i);
                    if (z) {
                        imageView.setImageDrawable(cVar);
                        return true;
                    }
                    imageView.setBackground(cVar);
                    return true;
                }
                return false;
            } catch (Resources.NotFoundException | IOException unused) {
            }
        }
        return false;
    }

    /* compiled from: GifViewUtils.java */
    /* loaded from: classes2.dex */
    public static class a extends b {

        /* renamed from: c  reason: collision with root package name */
        public final int f6246c;

        /* renamed from: d  reason: collision with root package name */
        public final int f6247d;

        public a(ImageView imageView, AttributeSet attributeSet, int i, int i2) {
            super(imageView, attributeSet, i, i2);
            this.f6246c = a(imageView, attributeSet, true);
            this.f6247d = a(imageView, attributeSet, false);
        }

        public static int a(ImageView imageView, AttributeSet attributeSet, boolean z) {
            int attributeResourceValue = attributeSet.getAttributeResourceValue("http://schemas.android.com/apk/res/android", z ? "src" : "background", 0);
            if (attributeResourceValue > 0) {
                if (g.f6245a.contains(imageView.getResources().getResourceTypeName(attributeResourceValue)) && !g.b(imageView, z, attributeResourceValue)) {
                    return attributeResourceValue;
                }
            }
            return 0;
        }

        public a() {
            this.f6246c = 0;
            this.f6247d = 0;
        }
    }

    /* compiled from: GifViewUtils.java */
    /* loaded from: classes2.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public boolean f6248a;

        /* renamed from: b  reason: collision with root package name */
        public final int f6249b;

        public b(View view, AttributeSet attributeSet, int i, int i2) {
            TypedArray obtainStyledAttributes = view.getContext().obtainStyledAttributes(attributeSet, i.f6251a, i, i2);
            this.f6248a = obtainStyledAttributes.getBoolean(0, false);
            this.f6249b = obtainStyledAttributes.getInt(1, -1);
            obtainStyledAttributes.recycle();
        }

        public b() {
            this.f6248a = false;
            this.f6249b = -1;
        }
    }
}