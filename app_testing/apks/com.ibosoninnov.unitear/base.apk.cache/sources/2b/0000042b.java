package b.j.b;

import android.app.Notification;
import android.app.PendingIntent;
import android.content.Context;
import android.os.Build;
import android.os.Bundle;
import java.util.ArrayList;
import java.util.Objects;

/* compiled from: NotificationCompat.java */
/* loaded from: classes.dex */
public class h {

    /* renamed from: a  reason: collision with root package name */
    public Context f2060a;

    /* renamed from: e  reason: collision with root package name */
    public CharSequence f2064e;

    /* renamed from: f  reason: collision with root package name */
    public CharSequence f2065f;

    /* renamed from: g  reason: collision with root package name */
    public PendingIntent f2066g;

    /* renamed from: h  reason: collision with root package name */
    public int f2067h;
    public i j;
    public Bundle l;
    public String m;
    public boolean n;
    public Notification o;
    @Deprecated
    public ArrayList<String> p;

    /* renamed from: b  reason: collision with root package name */
    public ArrayList<f> f2061b = new ArrayList<>();

    /* renamed from: c  reason: collision with root package name */
    public ArrayList<l> f2062c = new ArrayList<>();

    /* renamed from: d  reason: collision with root package name */
    public ArrayList<f> f2063d = new ArrayList<>();
    public boolean i = true;
    public boolean k = false;

    @Deprecated
    public h(Context context) {
        Notification notification = new Notification();
        this.o = notification;
        this.f2060a = context;
        this.m = null;
        notification.when = System.currentTimeMillis();
        this.o.audioStreamType = -1;
        this.f2067h = 0;
        this.p = new ArrayList<>();
        this.n = true;
    }

    public static CharSequence b(CharSequence charSequence) {
        return (charSequence != null && charSequence.length() > 5120) ? charSequence.subSequence(0, 5120) : charSequence;
    }

    public Notification a() {
        Notification build;
        Bundle bundle;
        j jVar = new j(this);
        i iVar = jVar.f2070b.j;
        if (iVar != null) {
            new Notification.BigTextStyle(jVar.f2069a).setBigContentTitle(null).bigText(((g) iVar).f2059b);
        }
        if (Build.VERSION.SDK_INT >= 26) {
            build = jVar.f2069a.build();
        } else {
            build = jVar.f2069a.build();
        }
        Objects.requireNonNull(jVar.f2070b);
        if (iVar != null) {
            Objects.requireNonNull(jVar.f2070b.j);
        }
        if (iVar != null && (bundle = build.extras) != null) {
            bundle.putString("androidx.core.app.extra.COMPAT_TEMPLATE", "androidx.core.app.NotificationCompat$BigTextStyle");
        }
        return build;
    }

    public final void c(int i, boolean z) {
        if (z) {
            Notification notification = this.o;
            notification.flags = i | notification.flags;
            return;
        }
        Notification notification2 = this.o;
        notification2.flags = (~i) & notification2.flags;
    }

    public h d(i iVar) {
        if (this.j != iVar) {
            this.j = iVar;
            if (iVar.f2068a != this) {
                iVar.f2068a = this;
                d(iVar);
            }
        }
        return this;
    }
}