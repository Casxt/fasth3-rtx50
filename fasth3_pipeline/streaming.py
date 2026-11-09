"""Ownership adapters for opt-in grouped streaming; baseline leaf path remains."""
import torch


class CPUWeightStore:
    """References an already CPU-loaded VAE, without cloning its weight storage.

    This is not used to materialize a full DiT/Qwen state_dict. It preserves the
    loaded VAE's dtype and legacy weight_norm keys until a file-backed VAE loader
    is separately validated. The caller must budget the existing CPU weights.
    """
    backend='preloaded_cpu_vae'

    def __init__(self,model):
        self.values=dict(model.state_dict())
        if any(not torch.is_tensor(t) or t.device.type!='cpu' or not t.is_contiguous() for t in self.values.values()):
            raise ValueError('CPUWeightStore requires initialized contiguous CPU-only tensor weights')
        self.weight_map={key:'preloaded CPU storage' for key in self.values}
        self.closed=False
        self.read_calls=self.read_bytes=0
        self.source_bytes=sum(t.numel()*t.element_size() for t in self.values.values())

    def tensor(self,key):
        if self.closed:raise RuntimeError('CPU weight store is closed')
        return self.values[key]

    def shape(self,key):return tuple(self.tensor(key).shape)
    def dtype(self,key):return self.tensor(key).dtype

    def validate_keys(self,model):
        expected=model.state_dict()
        if expected.keys()!=self.values.keys():raise ValueError('VAE CPU store architecture mismatch')
        for key,value in expected.items():
            if tuple(value.shape)!=self.shape(key):raise ValueError(f'VAE CPU store shape mismatch: {key}')

    def read_into(self,key,destination,*,byte_offset=0):
        if (destination.device.type!='cpu' or destination.dtype!=torch.uint8
                or not destination.is_contiguous() or type(byte_offset) is not int or byte_offset<0):
            raise ValueError('Requires a contiguous CPU uint8 destination and nonnegative byte offset')
        source=self.tensor(key).reshape(-1).view(torch.uint8)
        count=destination.numel()
        if byte_offset+count>source.numel():raise ValueError('CPU weight read exceeds tensor boundary')
        destination.reshape(-1).copy_(source[byte_offset:byte_offset+count])
        self.read_calls+=1;self.read_bytes+=count
        return count

    def stats_snapshot(self):
        return dict(backend=self.backend,source_payload_bytes=self.source_bytes,
                    read_into_calls=self.read_calls,read_into_bytes=self.read_bytes,closed=self.closed,
                    note='Existing VAE CPU storage references, not an extra cloned model; logical bytes, not disk I/O.')

    def close(self):
        self.values.clear();self.closed=True


class ManagedStream:
    """Close GPU consumers before releasing the owned backing store."""
    def __init__(self,stream,store,mode):
        self.stream,self.store,self.mode=stream,store,mode
        self.closed=False
        self._close_callbacks=[]

    def add_close_callback(self, callback):
        if self.closed:
            callback()
        else:
            self._close_callbacks.append(callback)

    def __getattr__(self,name):
        return getattr(self.stream,name)

    def stats_snapshot(self):
        return dict(mode=self.mode,closed=self.closed,
                    stream=self.stream.stats_snapshot() if hasattr(self.stream,'stats_snapshot') else {'profiling':'legacy leaf loader'},
                    storage=self.store.stats_snapshot() if hasattr(self.store,'stats_snapshot') else {'backend':'contextual_safe_open'})

    def _close_source(self):
        if hasattr(self.store,'close'):self.store.close()
        self.closed=True
        callbacks,self._close_callbacks=self._close_callbacks,[]
        for callback in callbacks:
            callback()

    def close(self):
        if self.closed:return
        try:
            self.stream.close()
        except BaseException:
            # Grouped stream sets closed even when genuine cleanup fails, but
            # leaves it False when refusing active/wrong-thread close. Preserve
            # its live backing store and ownership callback in that latter case.
            if getattr(self.stream,'closed',None) is False:
                raise
            self._close_source()
            raise
        else:
            self._close_source()
