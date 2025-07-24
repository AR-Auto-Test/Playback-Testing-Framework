package b.l.b;

import android.graphics.Rect;
import b.l.b.a;
import java.util.Comparator;

/* compiled from: FocusStrategy.java */
/* loaded from: classes.dex */
public class d<T> implements Comparator<T> {

    /* renamed from: b  reason: collision with root package name */
    public final Rect f2315b = new Rect();

    /* renamed from: c  reason: collision with root package name */
    public final Rect f2316c = new Rect();

    /* renamed from: d  reason: collision with root package name */
    public final boolean f2317d;

    /* renamed from: e  reason: collision with root package name */
    public final b<T> f2318e;

    public d(boolean z, b<T> bVar) {
        this.f2317d = z;
        this.f2318e = bVar;
    }

    @Override // java.util.Comparator
    public int compare(T t, T t2) {
        Rect rect = this.f2315b;
        Rect rect2 = this.f2316c;
        ((a.C0042a) this.f2318e).a(t, rect);
        ((a.C0042a) this.f2318e).a(t2, rect2);
        int i = rect.top;
        int i2 = rect2.top;
        if (i < i2) {
            return -1;
        }
        if (i > i2) {
            return 1;
        }
        int i3 = rect.left;
        int i4 = rect2.left;
        if (i3 < i4) {
            return this.f2317d ? 1 : -1;
        } else if (i3 > i4) {
            return this.f2317d ? -1 : 1;
        } else {
            int i5 = rect.bottom;
            int i6 = rect2.bottom;
            if (i5 < i6) {
                return -1;
            }
            if (i5 > i6) {
                return 1;
            }
            int i7 = rect.right;
            int i8 = rect2.right;
            if (i7 < i8) {
                return this.f2317d ? 1 : -1;
            } else if (i7 > i8) {
                return this.f2317d ? -1 : 1;
            } else {
                return 0;
            }
        }
    }
}