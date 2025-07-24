package c.a.a.y;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.text.TextUtils;
import android.view.View;
import c.a.a.c0.c;
import c.a.a.k;
import java.util.HashMap;
import java.util.Map;

/* compiled from: ImageAssetManager.java */
/* loaded from: classes.dex */
public class b {

    /* renamed from: a  reason: collision with root package name */
    public static final Object f3253a = new Object();

    /* renamed from: b  reason: collision with root package name */
    public final Context f3254b;

    /* renamed from: c  reason: collision with root package name */
    public String f3255c;

    /* renamed from: d  reason: collision with root package name */
    public c.a.a.b f3256d;

    /* renamed from: e  reason: collision with root package name */
    public final Map<String, k> f3257e;

    public b(Drawable.Callback callback, String str, c.a.a.b bVar, Map<String, k> map) {
        String str2;
        this.f3255c = str;
        if (!TextUtils.isEmpty(str)) {
            if (this.f3255c.charAt(str2.length() - 1) != '/') {
                this.f3255c += '/';
            }
        }
        if (!(callback instanceof View)) {
            c.b("LottieDrawable must be inside of a view for images to work.");
            this.f3257e = new HashMap();
            this.f3254b = null;
            return;
        }
        this.f3254b = ((View) callback).getContext();
        this.f3257e = map;
        this.f3256d = bVar;
    }

    public final Bitmap a(String str, Bitmap bitmap) {
        synchronized (f3253a) {
            this.f3257e.get(str).f3113e = bitmap;
        }
        return bitmap;
    }
}