package b.j.b;

import android.app.PendingIntent;
import android.graphics.drawable.Icon;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import androidx.core.graphics.drawable.IconCompat;
import java.lang.reflect.InvocationTargetException;

/* compiled from: NotificationCompat.java */
/* loaded from: classes.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public final Bundle f2051a;

    /* renamed from: b  reason: collision with root package name */
    public IconCompat f2052b;

    /* renamed from: c  reason: collision with root package name */
    public final m[] f2053c;

    /* renamed from: d  reason: collision with root package name */
    public final m[] f2054d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f2055e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f2056f;

    /* renamed from: g  reason: collision with root package name */
    public final int f2057g;

    /* renamed from: h  reason: collision with root package name */
    public final boolean f2058h;
    @Deprecated
    public int i;
    public CharSequence j;
    public PendingIntent k;

    /* JADX WARN: Removed duplicated region for block: B:24:0x008b  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public f(int i, CharSequence charSequence, PendingIntent pendingIntent) {
        IconCompat b2 = i == 0 ? null : IconCompat.b(null, "", i);
        Bundle bundle = new Bundle();
        this.f2056f = true;
        this.f2052b = b2;
        if (b2 != null) {
            int i2 = b2.f239b;
            if (i2 == -1) {
                int i3 = Build.VERSION.SDK_INT;
                Icon icon = (Icon) b2.f240c;
                if (i3 >= 28) {
                    i2 = icon.getType();
                } else {
                    try {
                        i2 = ((Integer) icon.getClass().getMethod("getType", new Class[0]).invoke(icon, new Object[0])).intValue();
                    } catch (IllegalAccessException e2) {
                        Log.e("IconCompat", "Unable to get icon type " + icon, e2);
                        i2 = -1;
                        if (i2 == 2) {
                        }
                        this.j = h.b(charSequence);
                        this.k = pendingIntent;
                        this.f2051a = bundle;
                        this.f2053c = null;
                        this.f2054d = null;
                        this.f2055e = true;
                        this.f2057g = 0;
                        this.f2056f = true;
                        this.f2058h = false;
                    } catch (NoSuchMethodException e3) {
                        Log.e("IconCompat", "Unable to get icon type " + icon, e3);
                        i2 = -1;
                        if (i2 == 2) {
                        }
                        this.j = h.b(charSequence);
                        this.k = pendingIntent;
                        this.f2051a = bundle;
                        this.f2053c = null;
                        this.f2054d = null;
                        this.f2055e = true;
                        this.f2057g = 0;
                        this.f2056f = true;
                        this.f2058h = false;
                    } catch (InvocationTargetException e4) {
                        Log.e("IconCompat", "Unable to get icon type " + icon, e4);
                        i2 = -1;
                        if (i2 == 2) {
                        }
                        this.j = h.b(charSequence);
                        this.k = pendingIntent;
                        this.f2051a = bundle;
                        this.f2053c = null;
                        this.f2054d = null;
                        this.f2055e = true;
                        this.f2057g = 0;
                        this.f2056f = true;
                        this.f2058h = false;
                    }
                }
            }
            if (i2 == 2) {
                this.i = b2.c();
            }
        }
        this.j = h.b(charSequence);
        this.k = pendingIntent;
        this.f2051a = bundle;
        this.f2053c = null;
        this.f2054d = null;
        this.f2055e = true;
        this.f2057g = 0;
        this.f2056f = true;
        this.f2058h = false;
    }

    public IconCompat a() {
        int i;
        if (this.f2052b == null && (i = this.i) != 0) {
            this.f2052b = IconCompat.b(null, "", i);
        }
        return this.f2052b;
    }
}