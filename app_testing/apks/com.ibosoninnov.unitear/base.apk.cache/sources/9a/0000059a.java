package b.v;

import android.os.Bundle;
import b.v.j;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/* compiled from: Navigator.java */
/* loaded from: classes.dex */
public abstract class q<D extends j> {

    /* compiled from: Navigator.java */
    /* loaded from: classes.dex */
    public interface a {
    }

    /* compiled from: Navigator.java */
    @Target({ElementType.TYPE})
    @Retention(RetentionPolicy.RUNTIME)
    /* loaded from: classes.dex */
    public @interface b {
        String value();
    }

    public abstract D a();

    public abstract j b(D d2, Bundle bundle, o oVar, a aVar);

    public void c(Bundle bundle) {
    }

    public Bundle d() {
        return null;
    }

    public abstract boolean e();
}