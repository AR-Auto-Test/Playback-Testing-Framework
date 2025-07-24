package b.b.d.a;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.drawable.Drawable;
import android.util.SparseArray;
import android.util.TypedValue;
import b.b.h.n0;
import java.util.WeakHashMap;

/* compiled from: AppCompatResources.java */
@SuppressLint({"RestrictedAPI"})
/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public static final ThreadLocal<TypedValue> f630a = new ThreadLocal<>();

    /* renamed from: b  reason: collision with root package name */
    public static final WeakHashMap<Context, SparseArray<?>> f631b = new WeakHashMap<>(0);

    /* renamed from: c  reason: collision with root package name */
    public static final Object f632c = new Object();

    public static Drawable a(Context context, int i) {
        return n0.c().e(context, i);
    }
}