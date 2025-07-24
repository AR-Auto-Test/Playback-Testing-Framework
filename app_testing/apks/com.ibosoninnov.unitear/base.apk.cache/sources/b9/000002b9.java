package b.d.a.e.y1.o;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.InputConfiguration;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.os.Build;
import b.d.a.e.y1.o.a;
import b.d.a.e.y1.o.b;
import b.d.a.e.y1.o.c;
import b.d.a.e.y1.o.d;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: SessionConfigurationCompat.java */
/* loaded from: classes.dex */
public final class g {

    /* renamed from: a  reason: collision with root package name */
    public final c f1338a;

    /* compiled from: SessionConfigurationCompat.java */
    /* loaded from: classes.dex */
    public static final class a implements c {

        /* renamed from: a  reason: collision with root package name */
        public final SessionConfiguration f1339a;

        /* renamed from: b  reason: collision with root package name */
        public final List<b.d.a.e.y1.o.b> f1340b;

        public a(int i, List<b.d.a.e.y1.o.b> list, Executor executor, CameraCaptureSession.StateCallback stateCallback) {
            b.a cVar;
            SessionConfiguration sessionConfiguration = new SessionConfiguration(i, g.f(list), executor, stateCallback);
            this.f1339a = sessionConfiguration;
            List<OutputConfiguration> outputConfigurations = sessionConfiguration.getOutputConfigurations();
            ArrayList arrayList = new ArrayList(outputConfigurations.size());
            for (OutputConfiguration outputConfiguration : outputConfigurations) {
                b.d.a.e.y1.o.b bVar = null;
                if (outputConfiguration != null) {
                    int i2 = Build.VERSION.SDK_INT;
                    if (i2 >= 28) {
                        cVar = new e(outputConfiguration);
                    } else if (i2 >= 26) {
                        cVar = new d(new d.a(outputConfiguration));
                    } else {
                        cVar = new b.d.a.e.y1.o.c(new c.a(outputConfiguration));
                    }
                    bVar = new b.d.a.e.y1.o.b(cVar);
                }
                arrayList.add(bVar);
            }
            this.f1340b = Collections.unmodifiableList(arrayList);
        }

        @Override // b.d.a.e.y1.o.g.c
        public b.d.a.e.y1.o.a a() {
            InputConfiguration inputConfiguration = this.f1339a.getInputConfiguration();
            if (inputConfiguration == null) {
                return null;
            }
            return new b.d.a.e.y1.o.a(new a.C0019a(inputConfiguration));
        }

        @Override // b.d.a.e.y1.o.g.c
        public CameraCaptureSession.StateCallback b() {
            return this.f1339a.getStateCallback();
        }

        @Override // b.d.a.e.y1.o.g.c
        public Object c() {
            return this.f1339a;
        }

        @Override // b.d.a.e.y1.o.g.c
        public int d() {
            return this.f1339a.getSessionType();
        }

        @Override // b.d.a.e.y1.o.g.c
        public List<b.d.a.e.y1.o.b> e() {
            return this.f1340b;
        }

        public boolean equals(Object obj) {
            if (obj instanceof a) {
                return Objects.equals(this.f1339a, ((a) obj).f1339a);
            }
            return false;
        }

        @Override // b.d.a.e.y1.o.g.c
        public void f(CaptureRequest captureRequest) {
            this.f1339a.setSessionParameters(captureRequest);
        }

        @Override // b.d.a.e.y1.o.g.c
        public Executor getExecutor() {
            return this.f1339a.getExecutor();
        }

        public int hashCode() {
            return this.f1339a.hashCode();
        }
    }

    /* compiled from: SessionConfigurationCompat.java */
    /* loaded from: classes.dex */
    public static final class b implements c {

        /* renamed from: a  reason: collision with root package name */
        public final List<b.d.a.e.y1.o.b> f1341a;

        /* renamed from: b  reason: collision with root package name */
        public final CameraCaptureSession.StateCallback f1342b;

        /* renamed from: c  reason: collision with root package name */
        public final Executor f1343c;

        /* renamed from: d  reason: collision with root package name */
        public int f1344d;

        public b(int i, List<b.d.a.e.y1.o.b> list, Executor executor, CameraCaptureSession.StateCallback stateCallback) {
            this.f1344d = i;
            this.f1341a = Collections.unmodifiableList(new ArrayList(list));
            this.f1342b = stateCallback;
            this.f1343c = executor;
        }

        @Override // b.d.a.e.y1.o.g.c
        public b.d.a.e.y1.o.a a() {
            return null;
        }

        @Override // b.d.a.e.y1.o.g.c
        public CameraCaptureSession.StateCallback b() {
            return this.f1342b;
        }

        @Override // b.d.a.e.y1.o.g.c
        public Object c() {
            return null;
        }

        @Override // b.d.a.e.y1.o.g.c
        public int d() {
            return this.f1344d;
        }

        @Override // b.d.a.e.y1.o.g.c
        public List<b.d.a.e.y1.o.b> e() {
            return this.f1341a;
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj instanceof b) {
                b bVar = (b) obj;
                Objects.requireNonNull(bVar);
                if (Objects.equals(null, null) && this.f1344d == bVar.f1344d && this.f1341a.size() == bVar.f1341a.size()) {
                    for (int i = 0; i < this.f1341a.size(); i++) {
                        if (!this.f1341a.get(i).equals(bVar.f1341a.get(i))) {
                            return false;
                        }
                    }
                    return true;
                }
            }
            return false;
        }

        @Override // b.d.a.e.y1.o.g.c
        public void f(CaptureRequest captureRequest) {
        }

        @Override // b.d.a.e.y1.o.g.c
        public Executor getExecutor() {
            return this.f1343c;
        }

        public int hashCode() {
            int hashCode = this.f1341a.hashCode() ^ 31;
            int i = ((hashCode << 5) - hashCode) ^ 0;
            return this.f1344d ^ ((i << 5) - i);
        }
    }

    /* compiled from: SessionConfigurationCompat.java */
    /* loaded from: classes.dex */
    public interface c {
        b.d.a.e.y1.o.a a();

        CameraCaptureSession.StateCallback b();

        Object c();

        int d();

        List<b.d.a.e.y1.o.b> e();

        void f(CaptureRequest captureRequest);

        Executor getExecutor();
    }

    public g(int i, List<b.d.a.e.y1.o.b> list, Executor executor, CameraCaptureSession.StateCallback stateCallback) {
        if (Build.VERSION.SDK_INT < 28) {
            this.f1338a = new b(i, list, executor, stateCallback);
        } else {
            this.f1338a = new a(i, list, executor, stateCallback);
        }
    }

    public static List<OutputConfiguration> f(List<b.d.a.e.y1.o.b> list) {
        ArrayList arrayList = new ArrayList(list.size());
        for (b.d.a.e.y1.o.b bVar : list) {
            arrayList.add((OutputConfiguration) bVar.f1330a.c());
        }
        return arrayList;
    }

    public Executor a() {
        return this.f1338a.getExecutor();
    }

    public b.d.a.e.y1.o.a b() {
        return this.f1338a.a();
    }

    public List<b.d.a.e.y1.o.b> c() {
        return this.f1338a.e();
    }

    public int d() {
        return this.f1338a.d();
    }

    public CameraCaptureSession.StateCallback e() {
        return this.f1338a.b();
    }

    public boolean equals(Object obj) {
        if (obj instanceof g) {
            return this.f1338a.equals(((g) obj).f1338a);
        }
        return false;
    }

    public int hashCode() {
        return this.f1338a.hashCode();
    }
}