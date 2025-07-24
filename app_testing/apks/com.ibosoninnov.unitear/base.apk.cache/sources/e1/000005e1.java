package b.z;

import android.view.View;
import android.view.WindowId;

/* compiled from: WindowIdApi18.java */
/* loaded from: classes.dex */
public class a0 implements b0 {

    /* renamed from: a  reason: collision with root package name */
    public final WindowId f2849a;

    public a0(View view) {
        this.f2849a = view.getWindowId();
    }

    public boolean equals(Object obj) {
        return (obj instanceof a0) && ((a0) obj).f2849a.equals(this.f2849a);
    }

    public int hashCode() {
        return this.f2849a.hashCode();
    }
}