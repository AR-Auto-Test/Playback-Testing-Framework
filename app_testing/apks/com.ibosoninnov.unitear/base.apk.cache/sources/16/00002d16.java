package f.g0.i;

/* compiled from: ErrorCode.java */
/* loaded from: classes2.dex */
public enum b {
    NO_ERROR(0),
    PROTOCOL_ERROR(1),
    INTERNAL_ERROR(2),
    FLOW_CONTROL_ERROR(3),
    REFUSED_STREAM(7),
    CANCEL(8),
    COMPRESSION_ERROR(9),
    CONNECT_ERROR(10),
    ENHANCE_YOUR_CALM(11),
    INADEQUATE_SECURITY(12),
    HTTP_1_1_REQUIRED(13);
    
    public final int n;

    b(int i) {
        this.n = i;
    }

    public static b a(int i) {
        b[] values = values();
        for (int i2 = 0; i2 < 11; i2++) {
            b bVar = values[i2];
            if (bVar.n == i) {
                return bVar;
            }
        }
        return null;
    }
}