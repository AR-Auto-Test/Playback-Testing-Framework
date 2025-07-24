package b.t;

import android.annotation.SuppressLint;
import android.app.Application;
import android.os.Bundle;
import androidx.lifecycle.SavedStateHandleController;
import com.google.firebase.crashlytics.internal.metadata.UserMetadata;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/* compiled from: SavedStateViewModelFactory.java */
/* loaded from: classes.dex */
public final class r extends v {

    /* renamed from: a  reason: collision with root package name */
    public static final Class<?>[] f2593a = {Application.class, q.class};

    /* renamed from: b  reason: collision with root package name */
    public static final Class<?>[] f2594b = {q.class};

    /* renamed from: c  reason: collision with root package name */
    public final Application f2595c;

    /* renamed from: d  reason: collision with root package name */
    public final t f2596d;

    /* renamed from: e  reason: collision with root package name */
    public final Bundle f2597e;

    /* renamed from: f  reason: collision with root package name */
    public final e f2598f;

    /* renamed from: g  reason: collision with root package name */
    public final b.x.a f2599g;

    @SuppressLint({"LambdaLast"})
    public r(Application application, b.x.c cVar, Bundle bundle) {
        this.f2599g = cVar.getSavedStateRegistry();
        this.f2598f = cVar.getLifecycle();
        this.f2597e = bundle;
        this.f2595c = application;
        if (t.f2602a == null) {
            t.f2602a = new t(application);
        }
        this.f2596d = t.f2602a;
    }

    public static <T> Constructor<T> d(Class<T> cls, Class<?>[] clsArr) {
        for (Constructor<?> constructor : cls.getConstructors()) {
            Constructor<T> constructor2 = (Constructor<T>) constructor;
            if (Arrays.equals(clsArr, constructor2.getParameterTypes())) {
                return constructor2;
            }
        }
        return null;
    }

    @Override // b.t.v, b.t.u
    public <T extends s> T a(Class<T> cls) {
        String canonicalName = cls.getCanonicalName();
        if (canonicalName != null) {
            return (T) c(canonicalName, cls);
        }
        throw new IllegalArgumentException("Local and anonymous classes can not be ViewModels");
    }

    @Override // b.t.x
    public void b(s sVar) {
        SavedStateHandleController.a(sVar, this.f2599g, this.f2598f);
    }

    @Override // b.t.v
    public <T extends s> T c(String str, Class<T> cls) {
        Constructor d2;
        q qVar;
        boolean isAssignableFrom = a.class.isAssignableFrom(cls);
        if (isAssignableFrom) {
            d2 = d(cls, f2593a);
        } else {
            d2 = d(cls, f2594b);
        }
        if (d2 == null) {
            return (T) this.f2596d.a(cls);
        }
        b.x.a aVar = this.f2599g;
        e eVar = this.f2598f;
        Bundle bundle = this.f2597e;
        Bundle a2 = aVar.a(str);
        int i = q.f2589a;
        if (a2 == null && bundle == null) {
            qVar = new q();
        } else {
            HashMap hashMap = new HashMap();
            if (bundle != null) {
                for (String str2 : bundle.keySet()) {
                    hashMap.put(str2, bundle.get(str2));
                }
            }
            if (a2 == null) {
                qVar = new q(hashMap);
            } else {
                ArrayList parcelableArrayList = a2.getParcelableArrayList(UserMetadata.KEYDATA_FILENAME);
                ArrayList parcelableArrayList2 = a2.getParcelableArrayList("values");
                if (parcelableArrayList != null && parcelableArrayList2 != null && parcelableArrayList.size() == parcelableArrayList2.size()) {
                    for (int i2 = 0; i2 < parcelableArrayList.size(); i2++) {
                        hashMap.put((String) parcelableArrayList.get(i2), parcelableArrayList2.get(i2));
                    }
                    qVar = new q(hashMap);
                } else {
                    throw new IllegalStateException("Invalid bundle passed as restored state");
                }
            }
        }
        SavedStateHandleController savedStateHandleController = new SavedStateHandleController(str, qVar);
        savedStateHandleController.b(aVar, eVar);
        SavedStateHandleController.g(aVar, eVar);
        try {
            T t = isAssignableFrom ? (T) d2.newInstance(this.f2595c, qVar) : (T) d2.newInstance(qVar);
            t.b("androidx.lifecycle.savedstate.vm.tag", savedStateHandleController);
            return t;
        } catch (IllegalAccessException e2) {
            throw new RuntimeException("Failed to access " + cls, e2);
        } catch (InstantiationException e3) {
            throw new RuntimeException("A " + cls + " cannot be instantiated.", e3);
        } catch (InvocationTargetException e4) {
            throw new RuntimeException("An exception happened in constructor of " + cls, e4.getCause());
        }
    }
}