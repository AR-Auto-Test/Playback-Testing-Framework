package c.d.c.i.b;

import com.google.firebase.encoders.ObjectEncoder;
import com.google.firebase.encoders.ObjectEncoderContext;
import com.google.firebase.encoders.proto.ProtobufDataEncoderContext;
import java.util.Map;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class a implements ObjectEncoder {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ a f4443a = new a();

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
    @Override // com.google.firebase.encoders.Encoder
    public final void encode(Object obj, ObjectEncoderContext objectEncoderContext) {
        ProtobufDataEncoderContext.lambda$static$0((Map.Entry) obj, objectEncoderContext);
    }
}