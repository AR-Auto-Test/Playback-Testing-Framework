package b.x;

import android.annotation.SuppressLint;
import android.os.Bundle;
import androidx.savedstate.Recreator;

/* compiled from: SavedStateRegistry.java */
@SuppressLint({"RestrictedApi"})
/* loaded from: classes.dex */
public final class a {

    /* renamed from: b  reason: collision with root package name */
    public Bundle f2821b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f2822c;

    /* renamed from: d  reason: collision with root package name */
    public Recreator.a f2823d;

    /* renamed from: a  reason: collision with root package name */
    public b.c.a.b.b<String, b> f2820a = new b.c.a.b.b<>();

    /* renamed from: e  reason: collision with root package name */
    public boolean f2824e = true;

    /* compiled from: SavedStateRegistry.java */
    /* renamed from: b.x.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public interface InterfaceC0055a {
        void a(c cVar);
    }

    /* compiled from: SavedStateRegistry.java */
    /* loaded from: classes.dex */
    public interface b {
        Bundle a();
    }

    public Bundle a(String str) {
        if (this.f2822c) {
            Bundle bundle = this.f2821b;
            if (bundle != null) {
                Bundle bundle2 = bundle.getBundle(str);
                this.f2821b.remove(str);
                if (this.f2821b.isEmpty()) {
                    this.f2821b = null;
                }
                return bundle2;
            }
            return null;
        }
        throw new IllegalStateException("You can consumeRestoredStateForKey only after super.onCreate of corresponding component");
    }

    public void b(Class<? extends InterfaceC0055a> cls) {
        if (this.f2824e) {
            if (this.f2823d == null) {
                this.f2823d = new Recreator.a(this);
            }
            try {
                cls.getDeclaredConstructor(new Class[0]);
                Recreator.a aVar = this.f2823d;
                aVar.f488a.add(cls.getName());
                return;
            } catch (NoSuchMethodException e2) {
                StringBuilder x = c.b.a.a.a.x("Class");
                x.append(cls.getSimpleName());
                x.append(" must have default constructor in order to be automatically recreated");
                throw new IllegalArgumentException(x.toString(), e2);
            }
        }
        throw new IllegalStateException("Can not perform this action after onSaveInstanceState");
    }
}