package c.a.a.y;

import android.content.res.AssetManager;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.view.View;
import c.a.a.c0.c;
import c.a.a.z.i;
import java.util.HashMap;
import java.util.Map;

/* compiled from: FontAssetManager.java */
/* loaded from: classes.dex */
public class a {

    /* renamed from: d  reason: collision with root package name */
    public final AssetManager f3251d;

    /* renamed from: a  reason: collision with root package name */
    public final i<String> f3248a = new i<>();

    /* renamed from: b  reason: collision with root package name */
    public final Map<i<String>, Typeface> f3249b = new HashMap();

    /* renamed from: c  reason: collision with root package name */
    public final Map<String, Typeface> f3250c = new HashMap();

    /* renamed from: e  reason: collision with root package name */
    public String f3252e = ".ttf";

    public a(Drawable.Callback callback) {
        if (!(callback instanceof View)) {
            c.b("LottieDrawable must be inside of a view for images to work.");
            this.f3251d = null;
            return;
        }
        this.f3251d = ((View) callback).getContext().getAssets();
    }
}