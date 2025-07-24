package b.q.b;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.IntentSender;
import android.os.Bundle;
import android.os.Handler;
import android.view.LayoutInflater;
import androidx.fragment.app.Fragment;

/* compiled from: FragmentHostCallback.java */
/* loaded from: classes.dex */
public abstract class n<E> extends j {

    /* renamed from: b  reason: collision with root package name */
    public final Activity f2489b;

    /* renamed from: c  reason: collision with root package name */
    public final Context f2490c;

    /* renamed from: d  reason: collision with root package name */
    public final Handler f2491d;

    /* renamed from: e  reason: collision with root package name */
    public final q f2492e;

    public n(d dVar) {
        Handler handler = new Handler();
        this.f2492e = new s();
        this.f2489b = dVar;
        b.j.b.d.h(dVar, "context == null");
        this.f2490c = dVar;
        b.j.b.d.h(handler, "handler == null");
        this.f2491d = handler;
    }

    public abstract void d(Fragment fragment);

    public abstract E e();

    public abstract LayoutInflater f();

    public abstract void g(Fragment fragment, String[] strArr, int i);

    public abstract boolean h(Fragment fragment);

    public abstract boolean i(String str);

    public abstract void j(Fragment fragment, @SuppressLint({"UnknownNullness"}) Intent intent, int i, Bundle bundle);

    public abstract void k(Fragment fragment, @SuppressLint({"UnknownNullness"}) IntentSender intentSender, int i, Intent intent, int i2, int i3, int i4, Bundle bundle);

    public abstract void l();
}