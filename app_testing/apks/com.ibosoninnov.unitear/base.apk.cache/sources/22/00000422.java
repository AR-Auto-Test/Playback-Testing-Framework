package b.j.b;

import android.app.Activity;
import android.app.Application;
import android.content.res.Configuration;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.List;

/* compiled from: ActivityRecreator.java */
/* loaded from: classes.dex */
public final class b {

    /* renamed from: a  reason: collision with root package name */
    public static final Class<?> f2031a;

    /* renamed from: b  reason: collision with root package name */
    public static final Field f2032b;

    /* renamed from: c  reason: collision with root package name */
    public static final Field f2033c;

    /* renamed from: d  reason: collision with root package name */
    public static final Method f2034d;

    /* renamed from: e  reason: collision with root package name */
    public static final Method f2035e;

    /* renamed from: f  reason: collision with root package name */
    public static final Method f2036f;

    /* renamed from: g  reason: collision with root package name */
    public static final Handler f2037g = new Handler(Looper.getMainLooper());

    /* compiled from: ActivityRecreator.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ c f2038b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ Object f2039c;

        public a(c cVar, Object obj) {
            this.f2038b = cVar;
            this.f2039c = obj;
        }

        @Override // java.lang.Runnable
        public void run() {
            this.f2038b.f2042b = this.f2039c;
        }
    }

    /* compiled from: ActivityRecreator.java */
    /* renamed from: b.j.b.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class RunnableC0032b implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Application f2040b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ c f2041c;

        public RunnableC0032b(Application application, c cVar) {
            this.f2040b = application;
            this.f2041c = cVar;
        }

        @Override // java.lang.Runnable
        public void run() {
            this.f2040b.unregisterActivityLifecycleCallbacks(this.f2041c);
        }
    }

    /* compiled from: ActivityRecreator.java */
    /* loaded from: classes.dex */
    public static final class c implements Application.ActivityLifecycleCallbacks {

        /* renamed from: b  reason: collision with root package name */
        public Object f2042b;

        /* renamed from: c  reason: collision with root package name */
        public Activity f2043c;

        /* renamed from: d  reason: collision with root package name */
        public final int f2044d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f2045e = false;

        /* renamed from: f  reason: collision with root package name */
        public boolean f2046f = false;

        /* renamed from: g  reason: collision with root package name */
        public boolean f2047g = false;

        public c(Activity activity) {
            this.f2043c = activity;
            this.f2044d = activity.hashCode();
        }

        @Override // android.app.Application.ActivityLifecycleCallbacks
        public void onActivityCreated(Activity activity, Bundle bundle) {
        }

        @Override // android.app.Application.ActivityLifecycleCallbacks
        public void onActivityDestroyed(Activity activity) {
            if (this.f2043c == activity) {
                this.f2043c = null;
                this.f2046f = true;
            }
        }

        /* JADX WARN: Code restructure failed: missing block: B:19:0x003d, code lost:
            r5.f2047g = true;
            r5.f2042b = null;
         */
        /* JADX WARN: Code restructure failed: missing block: B:20:0x0042, code lost:
            return;
         */
        @Override // android.app.Application.ActivityLifecycleCallbacks
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void onActivityPaused(Activity activity) {
            if (!this.f2046f || this.f2047g || this.f2045e) {
                return;
            }
            Object obj = this.f2042b;
            int i = this.f2044d;
            boolean z = false;
            try {
                Object obj2 = b.f2033c.get(activity);
                if (obj2 == obj && activity.hashCode() == i) {
                    b.f2037g.postAtFrontOfQueue(new b.j.b.c(b.f2032b.get(activity), obj2));
                    z = true;
                }
            } catch (Throwable th) {
                Log.e("ActivityRecreator", "Exception while fetching field values", th);
            }
        }

        @Override // android.app.Application.ActivityLifecycleCallbacks
        public void onActivityResumed(Activity activity) {
        }

        @Override // android.app.Application.ActivityLifecycleCallbacks
        public void onActivitySaveInstanceState(Activity activity, Bundle bundle) {
        }

        @Override // android.app.Application.ActivityLifecycleCallbacks
        public void onActivityStarted(Activity activity) {
            if (this.f2043c == activity) {
                this.f2045e = true;
            }
        }

        @Override // android.app.Application.ActivityLifecycleCallbacks
        public void onActivityStopped(Activity activity) {
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:24:0x0078 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:39:0x005d A[EXC_TOP_SPLITTER, SYNTHETIC] */
    static {
        Class<?> cls;
        Field field;
        Field field2;
        Method declaredMethod;
        Class<?> cls2;
        Method declaredMethod2;
        Class<?> cls3;
        Method method = null;
        try {
            cls = Class.forName("android.app.ActivityThread");
        } catch (Throwable unused) {
            cls = null;
        }
        f2031a = cls;
        try {
            field = Activity.class.getDeclaredField("mMainThread");
            field.setAccessible(true);
        } catch (Throwable unused2) {
            field = null;
        }
        f2032b = field;
        try {
            field2 = Activity.class.getDeclaredField("mToken");
            field2.setAccessible(true);
        } catch (Throwable unused3) {
            field2 = null;
        }
        f2033c = field2;
        Class<?> cls4 = f2031a;
        if (cls4 != null) {
            try {
                declaredMethod = cls4.getDeclaredMethod("performStopActivity", IBinder.class, Boolean.TYPE, String.class);
                declaredMethod.setAccessible(true);
            } catch (Throwable unused4) {
            }
            f2034d = declaredMethod;
            cls2 = f2031a;
            if (cls2 != null) {
                try {
                    declaredMethod2 = cls2.getDeclaredMethod("performStopActivity", IBinder.class, Boolean.TYPE);
                    declaredMethod2.setAccessible(true);
                } catch (Throwable unused5) {
                }
                f2035e = declaredMethod2;
                cls3 = f2031a;
                if (a() && cls3 != null) {
                    try {
                        Class<?> cls5 = Boolean.TYPE;
                        Method declaredMethod3 = cls3.getDeclaredMethod("requestRelaunchActivity", IBinder.class, List.class, List.class, Integer.TYPE, cls5, Configuration.class, Configuration.class, cls5, cls5);
                        declaredMethod3.setAccessible(true);
                        method = declaredMethod3;
                    } catch (Throwable unused6) {
                    }
                }
                f2036f = method;
            }
            declaredMethod2 = null;
            f2035e = declaredMethod2;
            cls3 = f2031a;
            if (a()) {
                Class<?> cls52 = Boolean.TYPE;
                Method declaredMethod32 = cls3.getDeclaredMethod("requestRelaunchActivity", IBinder.class, List.class, List.class, Integer.TYPE, cls52, Configuration.class, Configuration.class, cls52, cls52);
                declaredMethod32.setAccessible(true);
                method = declaredMethod32;
            }
            f2036f = method;
        }
        declaredMethod = null;
        f2034d = declaredMethod;
        cls2 = f2031a;
        if (cls2 != null) {
        }
        declaredMethod2 = null;
        f2035e = declaredMethod2;
        cls3 = f2031a;
        if (a()) {
        }
        f2036f = method;
    }

    public static boolean a() {
        int i = Build.VERSION.SDK_INT;
        return i == 26 || i == 27;
    }

    public static boolean b(Activity activity) {
        Object obj;
        if (Build.VERSION.SDK_INT >= 28) {
            activity.recreate();
            return true;
        } else if (a() && f2036f == null) {
            return false;
        } else {
            if (f2035e == null && f2034d == null) {
                return false;
            }
            try {
                Object obj2 = f2033c.get(activity);
                if (obj2 == null || (obj = f2032b.get(activity)) == null) {
                    return false;
                }
                Application application = activity.getApplication();
                c cVar = new c(activity);
                application.registerActivityLifecycleCallbacks(cVar);
                Handler handler = f2037g;
                handler.post(new a(cVar, obj2));
                if (a()) {
                    Method method = f2036f;
                    Boolean bool = Boolean.FALSE;
                    method.invoke(obj, obj2, null, null, 0, bool, null, null, bool, bool);
                } else {
                    activity.recreate();
                }
                handler.post(new RunnableC0032b(application, cVar));
                return true;
            } catch (Throwable unused) {
                return false;
            }
        }
    }
}