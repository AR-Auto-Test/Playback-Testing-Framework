package c.e.b.p000if;

import android.content.Context;
import android.content.SharedPreferences;

/* compiled from: AppPrefes.java */
/* renamed from: c.e.b.if.d  reason: invalid package */
/* loaded from: classes2.dex */
public class d {

    /* renamed from: a  reason: collision with root package name */
    public SharedPreferences f4871a;

    /* renamed from: b  reason: collision with root package name */
    public SharedPreferences.Editor f4872b;

    public d(Context context) {
        SharedPreferences sharedPreferences = context.getSharedPreferences("Unity", 0);
        this.f4871a = sharedPreferences;
        this.f4872b = sharedPreferences.edit();
    }
}